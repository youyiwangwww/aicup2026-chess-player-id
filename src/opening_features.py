"""Target-player opening heatmaps and grouped fingerprints."""
import numpy as np
import pandas as pd
from .sgf_parser import parse_target_moves


def game_heatmap(content, color, opening_moves=40, board_size=19):
    """Count target-player placements on a board-size square heatmap."""
    heatmap = np.zeros((board_size, board_size), dtype=np.float64)
    for row, column in parse_target_moves(content, color, opening_moves, board_size):
        heatmap[row, column] += 1
    return heatmap


def build_fingerprints(frame, group_column, opening_moves=40, board_size=19):
    """Average valid nonempty game heatmaps; log individual failures."""
    fingerprints, errors = {}, []
    for identity, group in frame.groupby(group_column, sort=True):
        total = np.zeros((board_size, board_size), dtype=np.float64)
        valid = 0
        for row in group.itertuples(index=False):
            try:
                heatmap = game_heatmap(row.sgf_content, row.color, opening_moves, board_size)
                if not heatmap.any():
                    raise ValueError('No target-player placements in opening')
                total += heatmap
                valid += 1
            except ValueError as exc:
                errors.append({'group_column': group_column, 'group_id': identity,
                               'game_id': row.game_id, 'error': str(exc)})
        if valid:
            fingerprints[identity] = (total / valid).ravel()
    return fingerprints, pd.DataFrame(errors, columns=['group_column', 'group_id', 'game_id', 'error'])
