"""Balanced exposure and opposite-color positives, sharing the unchanged encoder."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .metric_utils import choose_device, file_digest, seed_everything, write_json
from .phase26_common import embed_games, aggregate_games
from .phase28_common import encoder, metrics
from .train_triplet import save_checkpoint
from .utils import ROOT, write_csv


class ViewStore:
    """Reuse master cached arrays and attach TRAIN-only color metadata."""
    def __init__(self, master, ids, colors):
        self.master = master
        self.indices = [i for i, r in enumerate(master.records) if r['group_id'] in ids]
        self.records = [{**master.records[i], 'color': colors[master.records[i]['game_id']]} for i in self.indices]

    def __len__(self):
        return len(self.records)

    def features(self, index):
        return self.master.features(self.indices[index])


class ExposureTriplets(Dataset):
    """Exactly T anchor triplets/player/epoch, with interleaved player order."""
    def __init__(self, store, per_player=20, seed=42, cross_color=False, min_color_games=2):
        self.store, self.per_player, self.seed = store, per_player, seed
        self.cross_color = cross_color
        self.games, self.color_games = {}, {}
        for i, record in enumerate(store.records):
            player = record['group_id']
            self.games.setdefault(player, []).append(i)
            self.color_games.setdefault((player, record['color']), []).append(i)
        self.players = sorted(p for p, games in self.games.items() if len(games) >= 2 and
            (not cross_color or all(len(self.color_games.get((p, c), [])) >= min_color_games for c in ['B', 'W'])))
        if len(self.players) < 2 or per_player < 1:
            raise ValueError('Need >=2 valid TRAIN players and positive exposure budget')
        self.labels = {p: i for i, p in enumerate(self.players)}
        self.set_epoch(0)

    def __len__(self):
        return len(self.players) * self.per_player

    def set_epoch(self, epoch):
        """Permute player scheduling reproducibly without changing exposure counts."""
        self.epoch = epoch
        self.order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(len(self.players))

    def sample_indices(self, index):
        """Keep positives same-player/different-game; CrossColor never falls back."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        pi = int(self.order[index % len(self.players)])
        player = self.players[pi]
        if self.cross_color:
            color = 'B' if (index // len(self.players) + pi) % 2 == 0 else 'W'
            a = int(rng.choice(self.color_games[player, color]))
            p = int(rng.choice(self.color_games[player, 'W' if color == 'B' else 'B']))
        else:
            a, p = rng.choice(self.games[player], size=2, replace=False).tolist()
        ni = int(rng.integers(len(self.players)-1))
        ni += ni >= pi
        n = int(rng.choice(self.games[self.players[ni]]))
        return tuple((game, int(rng.integers(self.store.records[game]['num_positions']))) for game in [a, p, n])

    def __getitem__(self, index):
        """Return feature tensors and TRAIN labels for hard-negative masking only."""
        indices = self.sample_indices(index)
        arrays = tuple(torch.from_numpy(self.store.features(game)[position].astype(np.float32)) for game, position in indices)
        labels = torch.tensor([self.labels[self.store.records[game]['group_id']] for game, _ in indices])
        return (*arrays, labels)


def hard_negative(anchors, pool, anchor_labels, pool_labels):
    """Select nearest different-player embeddings, retaining selected gradients."""
    valid = anchor_labels[:, None] != pool_labels[None, :]
    if not valid.any(dim=1).all():
        raise ValueError('No different-player negative in batch')
    distances = torch.cdist(anchors.detach(), pool.detach())
    indices = distances.masked_fill(~valid, float('inf')).argmin(dim=1)
    return pool.index_select(0, indices), indices


def score_embeddings(model, candidates, queries, device, batch_size, players, questions):
    """Average positions and then games equally; return full candidate score vectors."""
    cg, qg = embed_games(model, candidates, device, batch_size), embed_games(model, queries, device, batch_size)
    c, q = aggregate_games(cg, candidates.records), aggregate_games(qg, queries.records)
    if set(c) != set(players) or set(q) != set(questions):
        raise ValueError('Feature failures removed a DEV2 player/question; cannot silently change pool')
    scores = np.stack([q[key] for key in questions]) @ np.stack([c[key] for key in players]).T
    return scores, cg, qg


def diagnostics(cg, qg, candidate_records, query_records, truth, settings):
    """Summarize game cosine separation and dimension-wise variance of unit game embeddings."""
    c = cg / np.linalg.norm(cg, axis=1, keepdims=True)
    q = qg / np.linalg.norm(qg, axis=1, keepdims=True)
    mapping = dict(zip(truth.question_id, truth.player_id))
    labels_c = np.array([r['group_id'] for r in candidate_records])
    labels_q = np.array([mapping[r['group_id']] for r in query_records])
    mask = labels_c[:, None] == labels_q[None, :]
    similarities = c @ q.T
    output = {}
    for prefix, values in [('same', similarities[mask]), ('different', similarities[~mask])]:
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Invalid diagnostic game pairs')
        output.update({prefix + '_' + key: float(value) for key, value in [
            ('mean', values.mean()), ('median', np.median(values)), ('std', values.std()),
            ('p25', np.percentile(values, 25)), ('p75', np.percentile(values, 75))]})
    variance = c.var(axis=0)
    output.update(separation=output['same_mean']-output['different_mean'], variance_mean=float(variance.mean()),
                  variance_median=float(np.median(variance)), variance_min=float(variance.min()), variance_max=float(variance.max()),
                  near_zero_fraction=float(np.mean(variance < settings['variance_threshold'])))
    output['collapse_warning'] = output['near_zero_fraction'] >= settings['collapse_fraction']
    return output, variance


def train_experiment(config, name, store, candidates, queries, truth, players, questions, cross_color=False):
    """Train a fresh controlled model, selecting only the highest DEV2 score."""
    directory = ROOT / config['output_dir'] / 'experiments' / name
    dataset = ExposureTriplets(store, config['training']['triplets_per_player_per_epoch'], config['seed'],
                              cross_color, config['training']['cross_color_min_games'])
    if set(dataset.players) != {r['group_id'] for r in store.records}:
        raise ValueError('Filter CrossColor eligibility explicitly before training; no silent exclusions')
    identity = {'config': config, 'name': name, 'players': dataset.players, 'cross_color': cross_color,
                'source_hash': file_digest(ROOT / config['output_dir'] / 'splits/train.csv'), 'torch': str(torch.__version__),
                'validation_hashes': {key: file_digest(config['paths'][key]) for key in
                                      ['val_candidates_csv', 'val_queries_csv', 'val_ground_truth_csv']},
                'pool_implementation': 'deterministic_adaptive_window_mean_v1'}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state_path = directory / 'state.json'
    device = choose_device(config['training']['device'])
    seed_everything(config['seed'], config['training']['deterministic'], config['training']['num_threads'])
    model = encoder(config, device)
    state = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['signature'] != signature or file_digest(directory / 'best.pt') != state['checkpoint_sha256']:
            raise ValueError('Existing experiment provenance changed')
        print(f'{name}: reuse completed DEV2 run', flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        criterion = torch.nn.TripletMarginLoss(margin=config['training']['margin'], p=2)
        best, history = -1.0, []
        print(f'{name}: {len(dataset.players)} players, {len(dataset)} triplets/epoch, device={device}', flush=True)
        for epoch in range(1, config['training']['epochs']+1):
            started = time.perf_counter()
            dataset.set_epoch(epoch-1)
            model.train()
            total, seen = 0.0, 0
            exposure = {p: 0 for p in dataset.players}
            for a, p, n, labels in DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0):
                batch = len(a)
                pool = model(torch.cat([a, p, n]).to(device))
                aa, pp, _ = pool.split(batch)
                labels = labels.to(device)
                nn, _ = hard_negative(aa, pool, labels[:, 0], labels.T.reshape(-1))
                loss = criterion(aa, pp, nn)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite triplet loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * batch
                seen += batch
                for label in labels[:, 0].cpu().tolist():
                    exposure[dataset.players[label]] += 1
            if set(exposure.values()) != {dataset.per_player} or seen != len(dataset):
                raise ValueError('Actual per-player exposure differs from protocol')
            scores, _, _ = score_embeddings(model, candidates, queries, device, config['inference']['batch_size'], players, questions)
            values, _ = metrics(scores, players, questions, truth)
            improved = values['competition_score'] > best
            row = {'epoch': epoch, 'train_loss': total/seen, **values, 'is_best': improved,
                   'samples_per_epoch': seen, 'min_anchor_exposure': min(exposure.values()),
                   'max_anchor_exposure': max(exposure.values()), 'elapsed_seconds': time.perf_counter()-started}
            history.append(row)
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'model_config': config['model'],
                          'feature_config': config['features'], 'seed': config['seed'], 'signature': signature,
                          'cross_color_positive': cross_color, 'training_players': dataset.players, 'validation_metrics': values,
                          'pool_implementation': identity['pool_implementation']}
            save_checkpoint(checkpoint, directory / 'last.pt')
            if improved:
                best = values['competition_score']
                save_checkpoint(checkpoint, directory / 'best.pt')
            write_csv(pd.DataFrame(history), directory / 'training_log.csv')
            print(f'{name} epoch={epoch} loss={row["train_loss"]:.6f} DEV2={values["competition_score"]:.6f} best={improved} elapsed={row["elapsed_seconds"]:.1f}s', flush=True)
        best_row = max(history, key=lambda r: r['competition_score'])
        state = {'signature': signature, 'best_epoch': best_row['epoch'], 'metrics': {k: best_row[k] for k in values},
                 'players': dataset.players, 'samples_per_epoch': len(dataset), 'cross_color_positive': cross_color,
                 'checkpoint_sha256': file_digest(directory / 'best.pt')}
        write_json(state, state_path)
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    scores, cg, qg = score_embeddings(model, candidates, queries, device, config['inference']['batch_size'], players, questions)
    diag, variance = diagnostics(cg, qg, candidates.records, queries.records, truth, config['diagnostics'])
    write_csv(pd.DataFrame({'dimension': np.arange(len(variance)), 'variance': variance}), directory / 'dimension_variance.csv')
    np.savez_compressed(directory / 'dev_scores.npz', scores=scores, players=np.array(players), questions=np.array(questions))
    row = {'experiment': name, 'train_players': len(dataset.players), 'games_per_player': config['split']['train_games'],
           'samples_per_epoch': len(dataset), 'triplets_per_player': dataset.per_player, 'epochs': config['training']['epochs'],
           'best_epoch': state['best_epoch'], **state['metrics'], 'cross_color_positive': cross_color}
    return row, scores, {'experiment': name, **diag}, directory / 'best.pt'
