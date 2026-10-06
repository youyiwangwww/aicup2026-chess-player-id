"""Independent DEV-only helpers; preserve Phase 2.5 and its closed TEST."""
import contextlib
import contextvars
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .baseline import predict
from .feature_cache import GameFeatureStore
from .metric_utils import file_digest, write_json
from .utils import ROOT, read_csv, resolve_path

_guard = contextvars.ContextVar('phase26_read_guard', default=False)


def _audit_open(event, args):
    """Catch actual Python file opens, including pandas and pathlib APIs."""
    if event == 'open' and _guard.get() and isinstance(args[0], (str, bytes, Path)):
        name = '/' + str(args[0]).replace('\\', '/').lower().lstrip('/')
        if ('final_test2' in name or '/outputs/round1_archive/' in name
                or '/outputs/splits/test_' in name):
            raise PermissionError('Phase 2.6 DEV tuning may not open final/closed TEST files')


sys.addaudithook(_audit_open)


@contextlib.contextmanager
def dev_only():
    """Reject FINAL TEST 2 and closed Round 1 TEST access during DEV work."""
    token = _guard.set(True)
    try:
        yield
    finally:
        _guard.reset(token)


def dev_frames(config):
    """Read only audited DEV partitions, preserving anonymous query CSVs."""
    from .build_metric_split import read_training_validation
    return read_training_validation(config)


def variant_config(base, name, players, games, positions, epochs, strategy='random', original=False):
    """Isolate artifacts for one predetermined ablation without touching Round 1."""
    config = copy.deepcopy(base)
    directory = f'outputs/phase26/experiments/{name}'
    paths = config['paths']
    paths.update(metric_train_csv=f'{directory}/train.csv',
                 split_audit_json=f'{directory}/split_audit.json',
                 best_checkpoint=f'{directory}/best.pt', last_checkpoint=f'{directory}/last.pt',
                 training_state_json=f'{directory}/training_state.json',
                 training_log_csv=f'{directory}/training_log.csv',
                 feature_errors_csv=f'{directory}/feature_errors.csv',
                 feature_coverage_csv=f'{directory}/feature_coverage.csv')
    config['experiment'] = name
    config['output_dir'] = directory
    config['split'].update(train_players=players, train_games_per_player=games)
    config['features']['max_positions_per_game'] = positions
    config['training'].update(epochs=epochs, triplet_strategy=strategy)
    if original:
        config['model'].update(channels=32, num_blocks=4, embedding_dim=64)
    return config


class MemoryStore:
    """Load bounded uint8 DEV features once, avoiding random shard decompression."""
    def __init__(self, config, partition):
        from .feature_cache import load_store
        store = load_store(config, partition)
        self.records = store.records
        self.manifest = store.manifest
        # Group shard reads; preserve original record order and equal game weights.
        self.arrays = [None] * len(store)
        for index in sorted(range(len(store)), key=lambda i: store.records[i]['shard']):
            self.arrays[index] = store.features(index).copy()

    def __len__(self):
        return len(self.records)

    def features(self, index):
        """Return the same uint8 tensor as the original lazy feature store."""
        return self.arrays[index]


def embed_games(model, store, device, batch_size=64):
    """Batch across games, retaining exactly the existing position/game averaging."""
    model.eval()
    sums = [np.zeros(model.projection.out_features, dtype=np.float64) for _ in store.records]
    counts = [0] * len(store)
    pending, owners = [], []

    def flush():
        if not pending:
            return
        tensor = torch.from_numpy(np.stack(pending).astype(np.float32)).to(device)
        embeddings = model(tensor).cpu().numpy()
        for owner, embedding in zip(owners, embeddings):
            sums[owner] += embedding
            counts[owner] += 1
        pending.clear()
        owners.clear()

    with torch.inference_mode():
        for index in range(len(store)):
            for feature in store.features(index):
                pending.append(feature)
                owners.append(index)
                if len(pending) == batch_size:
                    flush()
        flush()
    return np.stack([value / count for value, count in zip(sums, counts)])


def aggregate_games(games, records):
    """Average games equally and normalize each identity/question embedding."""
    groups = {}
    for embedding, record in zip(games, records):
        groups.setdefault(record['group_id'], []).append(embedding)
    result = {}
    for identity, values in sorted(groups.items()):
        mean = np.mean(values, axis=0)
        norm = np.linalg.norm(mean)
        if norm <= 1e-12 or not np.isfinite(norm):
            raise ValueError(f'Invalid embedding for {identity}')
        result[identity] = mean / norm
    return result


def dev_retrieve(model, candidates, queries, device, batch_size=64):
    """Retrieve DEV identities with game-balanced batched inference."""
    cg = embed_games(model, candidates, device, batch_size)
    qg = embed_games(model, queries, device, batch_size)
    return predict(aggregate_games(cg, candidates.records), aggregate_games(qg, queries.records), 5), cg, qg
