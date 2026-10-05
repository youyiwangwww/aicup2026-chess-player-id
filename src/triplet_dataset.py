"""Deterministic cross-game positives and cross-player negatives."""
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """Sample games uniformly within players, then positions within games."""

    def __init__(self, store, samples_per_epoch=1024, seed=42):
        if samples_per_epoch < 1 or seed < 0:
            raise ValueError('samples_per_epoch must be positive and seed nonnegative')
        self.store = store
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        grouped = defaultdict(list)
        for index, record in enumerate(store.records):
            grouped[record['group_id']].append(index)
        self.games = {p: indices for p, indices in sorted(grouped.items()) if len(indices) >= 2}
        self.players = sorted(self.games)
        self.excluded_players = sorted(set(grouped) - set(self.players))
        if len(self.players) < 2:
            raise ValueError('Triplet dataset needs >=2 valid players with >=2 valid games each; '
                             'inspect feature_errors_csv or increase training games')
        for player in self.players:
            ids = [store.records[i]['game_id'] for i in self.games[player]]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate game IDs cannot form cross-game positives')

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        """Use a different reproducible triplet sequence each epoch."""
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch = epoch

    def sample_indices(self, index):
        """Return anchor/positive/negative (game index, position index) pairs."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        player_index = int(rng.integers(len(self.players)))
        player = self.players[player_index]
        a, p = rng.choice(self.games[player], size=2, replace=False).tolist()
        other_indices = [i for i in range(len(self.players)) if i != player_index]
        negative_player = self.players[int(rng.choice(other_indices))]
        n = int(rng.choice(self.games[negative_player]))
        return tuple((game, int(rng.integers(self.store.records[game]['num_positions'])))
                     for game in (a, p, n))

    def __getitem__(self, index):
        """Return float32 tensors only; evaluation labels never enter this dataset."""
        return tuple(torch.from_numpy(self.store.features(game)[position].astype(np.float32))
                     for game, position in self.sample_indices(index))
