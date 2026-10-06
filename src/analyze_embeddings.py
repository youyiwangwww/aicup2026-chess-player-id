"""Quantify same/different-player cosine separation on DEV games without GUI."""
import argparse

import numpy as np
import pandas as pd

from .embed_players import load_encoder
from .metric_utils import choose_device, seed_everything, write_json
from .phase26_common import MemoryStore, dev_frames, dev_only, embed_games
from .utils import ROOT, load_config, write_csv


def similarity_statistics(candidate_games, query_games, candidate_labels, query_labels):
    """Compare all candidate/query game pairs, normalizing each game before cosine."""
    cn = np.linalg.norm(candidate_games, axis=1, keepdims=True)
    qn = np.linalg.norm(query_games, axis=1, keepdims=True)
    if not np.isfinite(cn).all() or not np.isfinite(qn).all() or (cn <= 1e-12).any() or (qn <= 1e-12).any():
        raise ValueError('Cannot compute cosine similarity for zero/nonfinite game embeddings')
    c = candidate_games / cn
    q = query_games / qn
    similarities = c @ q.T
    same = np.asarray(candidate_labels)[:, None] == np.asarray(query_labels)[None, :]
    rows = []
    for name, mask in [('same_player', same), ('different_player', ~same)]:
        values = similarities[mask]
        if not len(values):
            raise ValueError('Need both same-player and different-player game pairs')
        rows.append({'pair_type': name, 'pair_count': len(values), 'mean': float(values.mean()),
                     'median': float(np.median(values)), 'std': float(values.std(ddof=0)),
                     'p25': float(np.quantile(values, .25)), 'p75': float(np.quantile(values, .75))})
    separation = rows[0]['mean'] - rows[1]['mean']
    interval_overlap = max(rows[0]['p25'], rows[1]['p25']) <= min(rows[0]['p75'], rows[1]['p75'])
    pooled_std = np.sqrt((rows[0]['std'] ** 2 + rows[1]['std'] ** 2) / 2)
    effect = separation / pooled_std if pooled_std else 0.0
    conclusion = ('Same/different IQRs overlap and standardized separation is small; '
                  'embedding has not learned clear player separability.'
                  if interval_overlap and effect < .5 else
                  'Inspect separation together with DEV retrieval; pair statistics are descriptive, not independent samples.')
    return pd.DataFrame(rows), {'separation': separation, 'iqr_overlap': bool(interval_overlap),
                               'standardized_separation': float(effect), 'interpretation': conclusion}


def analyze_embeddings(config):
    """Load selected DEV checkpoint and report per-game cosine distributions."""
    with dev_only():
        seed_everything(config['seed'], True, config['training']['num_threads'])
        device = choose_device(config['training']['device'])
        _, _, _, truth = dev_frames(config)
        model, _ = load_encoder(config['paths']['best_checkpoint'], config, device)
        candidates, queries = MemoryStore(config, 'val_candidate'), MemoryStore(config, 'val_query')
        cg = embed_games(model, candidates, device, config['inference']['batch_size'])
        qg = embed_games(model, queries, device, config['inference']['batch_size'])
        identities = truth.set_index('question_id').player_id.to_dict()
        statistics, summary = similarity_statistics(cg, qg,
            [r['group_id'] for r in candidates.records], [identities[r['group_id']] for r in queries.records])
        write_csv(statistics, 'outputs/phase26/embedding_similarity.csv')
        write_json(summary, 'outputs/phase26/embedding_diagnostics.json')
        print(statistics.to_string(index=False), flush=True)
        print(summary, flush=True)
        return statistics, summary


def main():
    """Inspect the selected configuration without reading any TEST ground truth."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/phase26_selected.yaml'))
    analyze_embeddings(load_config(parser.parse_args().config))


if __name__ == '__main__':
    main()
