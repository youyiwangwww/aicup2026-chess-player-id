"""Isolated DEV training with random or in-batch hard negatives."""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import evaluate
from .experiment_state import experiment_signature
from .metric_utils import choose_device, file_digest, seed_everything, write_json
from .phase26_common import MemoryStore, dev_frames, dev_only, dev_retrieve
from .player_model import PlayerEncoder
from .train_triplet import save_checkpoint
from .triplet_dataset import TripletDataset
from .utils import write_csv


class LabeledTriplets(TripletDataset):
    """Attach TRAIN identities to sampled positions solely for negative masking."""
    def __getitem__(self, index):
        indices = self.sample_indices(index)
        tensors = tuple(torch.from_numpy(self.store.features(game)[position].astype(np.float32))
                        for game, position in indices)
        labels = torch.tensor([self.players.index(self.store.records[game]['group_id'])
                               for game, _ in indices], dtype=torch.long)
        return (*tensors, labels)


def nearest_other_player(anchors, pool, anchor_labels, pool_labels):
    """Find each anchor's nearest batch embedding from a different TRAIN player."""
    distances = torch.cdist(anchors.detach(), pool.detach(), p=2)
    valid = anchor_labels[:, None] != pool_labels[None, :]
    if not valid.any(dim=1).all():
        raise ValueError('Every hard anchor needs a different-player batch negative')
    indices = distances.masked_fill(~valid, float('inf')).argmin(dim=1)
    return pool[indices], indices


def train_dev(config):
    """Select only by DEV VAL score and save independently resumable experiment outputs."""
    with dev_only():
        settings = config['training']
        seed_everything(config['seed'], settings['deterministic'], settings['num_threads'])
        device = choose_device(settings['device'])
        _, _, _, truth = dev_frames(config)
        training = MemoryStore(config, 'train')
        candidates = MemoryStore(config, 'val_candidate')
        queries = MemoryStore(config, 'val_query')
        dataset = LabeledTriplets(training, settings['samples_per_epoch'], config['seed'])
        model = PlayerEncoder(**config['model']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                      weight_decay=settings['weight_decay'])
        criterion = nn.TripletMarginLoss(margin=settings['margin'], p=2)
        strategy = settings.get('triplet_strategy', 'random')
        if strategy not in ('random', 'batch_hard'):
            raise ValueError('Unknown triplet strategy')
        best, history = -1.0, []
        print(f'{config["experiment"]}: device={device}, players={len(dataset.players)}, '
              f'games={len(training)}, positions={config["features"]["max_positions_per_game"]}, '
              f'epochs={settings["epochs"]}, strategy={strategy}', flush=True)
        for epoch in range(1, settings['epochs'] + 1):
            started = time.perf_counter()
            dataset.set_epoch(epoch - 1)
            loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=False,
                                num_workers=0, generator=torch.Generator().manual_seed(config['seed'] + epoch))
            model.train()
            total, samples = 0.0, 0
            for anchor, positive, negative, labels in loader:
                batch = len(anchor)
                pool = model(torch.cat([anchor, positive, negative]).to(device))
                a, p, n = pool.split(batch)
                if strategy == 'batch_hard':
                    labels = labels.to(device)
                    n, _ = nearest_other_player(a, pool, labels[:, 0], labels.T.reshape(-1))
                loss = criterion(a, p, n)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite triplet loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * batch
                samples += batch
            predictions, _, _ = dev_retrieve(model, candidates, queries, device, config['inference']['batch_size'])
            metrics, _ = evaluate(predictions, truth)
            improved = metrics['competition_score'] > best
            if improved:
                best = metrics['competition_score']
            row = {'epoch': epoch, 'train_loss': total / samples,
                   'val_top1': metrics['top_1_accuracy'], 'val_top3': metrics['top_3_accuracy'],
                   'val_top5': metrics['top_5_accuracy'], 'val_score': metrics['competition_score'],
                   'is_best': improved, 'elapsed_seconds': time.perf_counter() - started}
            history.append(row)
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                          'model_config': config['model'], 'feature_config': config['features'],
                          'feature_seed': config['seed'], 'validation_metrics': metrics,
                          'train_source_sha256': file_digest(config['paths']['metric_train_csv']),
                          'experiment': config['experiment'], 'training_config': settings}
            save_checkpoint(checkpoint, config['paths']['last_checkpoint'])
            if improved:
                save_checkpoint(checkpoint, config['paths']['best_checkpoint'])
            write_csv(pd.DataFrame(history), config['paths']['training_log_csv'])
            print(f'{config["experiment"]} epoch={epoch} loss={row["train_loss"]:.6f} '
                  f'VAL score={row["val_score"]:.6f} best={improved} '
                  f'elapsed={row["elapsed_seconds"]:.1f}s', flush=True)
        best_row = max(history, key=lambda row: row['val_score'])
        state = {'status': 'completed', 'best_epoch': best_row['epoch'],
                 'experiment_signature': experiment_signature(config),
                 'best_validation_score': best_row['val_score'],
                 'best_checkpoint_sha256': file_digest(config['paths']['best_checkpoint']),
                 'test_evaluations': 0, 'selection': 'DEV VAL competition score only'}
        write_json(state, config['paths']['training_state_json'])
        return best_row
