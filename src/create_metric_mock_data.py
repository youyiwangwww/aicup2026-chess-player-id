"""Create a separate synthetic Go dataset with unseen evaluation players."""
import argparse

import numpy as np
import pandas as pd

from .utils import ROOT, write_csv


def metric_mock_frame(num_players=10, games_per_player=6, seed=42):
    """Build valid alternating-color games with different board-region preferences."""
    if not 2 <= num_players <= 18 or games_per_player < 2:
        raise ValueError('Mock generator supports 2..18 players and >=2 games each')
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(num_players):
        # Distinct two-column bands; positions vary by game, unlike exact-copy SGFs.
        band = (player * 2) % 18
        own_points = [(col, row) for col in [band, band + 1] for row in range(15)]
        opponent_points = [(18, row) for row in range(15)]
        for game in range(games_per_player):
            color = 'B' if game % 2 == 0 else 'W'
            own_order = rng.permutation(len(own_points))[:10]
            other_order = rng.permutation(len(opponent_points))[:10]
            moves = []
            for own_index, other_index in zip(own_order, other_order):
                own_col, own_row = own_points[own_index]
                other_col, other_row = opponent_points[other_index]
                own = chr(97 + own_col) + chr(97 + own_row)
                other = chr(97 + other_col) + chr(97 + other_row)
                black, white = (own, other) if color == 'B' else (other, own)
                moves.append(f';B[{black}];W[{white}]')
            content = f'(;GM[1]FF[4]SZ[19]C[metric-mock-{player}-{game}]' + ''.join(moves) + ')'
            rows.append({'player_id': f'style_{player:02d}', 'game_id': f'metric_g_{player}_{game}',
                         'rank': 'A', 'color': color, 'sgf_content': content})
    return pd.DataFrame(rows)


def main():
    """Write Phase 2 mock data without touching the original Phase 1 fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--players', type=int, default=10)
    parser.add_argument('--games-per-player', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=str(ROOT / 'data/mock/metric_train_A.csv'))
    args = parser.parse_args()
    frame = metric_mock_frame(args.players, args.games_per_player, args.seed)
    path = args.output
    write_csv(frame, path)
    print(f'Created {len(frame)} synthetic Go games at {path}', flush=True)


if __name__ == '__main__':
    main()
