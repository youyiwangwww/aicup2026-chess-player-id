"""Lazy game access with a bounded LRU of compressed feature shards."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from .metric_utils import file_digest
from .player_features import FEATURE_VERSION
from .utils import resolve_path


class GameFeatureStore:
    """Expose game records without loading the whole feature collection into RAM."""

    def __init__(self, manifest_path, max_cached_shards=2):
        if max_cached_shards < 1:
            raise ValueError('max_cached_shards must be positive')
        with resolve_path(manifest_path).open(encoding='utf-8') as stream:
            self.manifest = json.load(stream)
        self.records = self.manifest['records']
        self.directory = Path(self.manifest['cache_directory'])
        self.max_cached_shards = max_cached_shards
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def features(self, index):
        """Decompress one shard when needed and return a game's uint8 positions."""
        record = self.records[index]
        name = record['shard']
        if name not in self._cache:
            with np.load(self.directory / name, allow_pickle=False) as shard:
                arrays = shard['features'], shard['offsets']
            self._cache[name] = arrays
            while len(self._cache) > self.max_cached_shards:
                self._cache.popitem(last=False)
        self._cache.move_to_end(name)
        features, offsets = self._cache[name]
        slot = record['slot']
        return features[offsets[slot]:offsets[slot + 1]]


def load_store(config, partition):
    """Reject cached features from an older CSV, feature scheme or sampling seed."""
    source_key, manifest_key, column = {
        'train': ('metric_train_csv', 'train_manifest', 'player_id'),
        'candidate': ('candidates_csv', 'candidate_manifest', 'player_id'),
        'query': ('queries_csv', 'query_manifest', 'question_id'),
    }[partition]
    store = GameFeatureStore(config['paths'][manifest_key], config['cache']['max_cached_shards'])
    manifest = store.manifest
    if (manifest['source_sha256'] != file_digest(config['paths'][source_key])
            or manifest['features'] != config['features']
            or manifest['seed'] != config['seed']
            or manifest['feature_version'] != FEATURE_VERSION
            or manifest['group_column'] != column):
        raise ValueError(f'Stale {partition} cache; rerun python -m src.player_features with this config')
    if not len(store):
        raise ValueError(f'No valid {partition} games; inspect feature_errors_csv')
    return store
