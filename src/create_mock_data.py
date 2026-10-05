"""Create small synthetic CSVs for a complete local smoke test."""
import pandas as pd
from .utils import ROOT, write_csv


def create_mock_data():
    """Generate six distinct opening styles, alternating target colors."""
    rows = []
    for player in range(6):
        column = 'abcdef'[player]
        for game in range(8):
            color = 'B' if game % 2 == 0 else 'W'
            moves = []
            for turn in range(12):
                own = f'{column}{chr(97 + turn)}'
                opponent = f's{chr(97 + turn)}'
                black, white = (own, opponent) if color == 'B' else (opponent, own)
                moves.append(f';B[{black}];W[{white}]')
            sgf = f'(;GM[1]FF[4]SZ[19]C[mock-{player}-{game}]' + ''.join(moves) + ')'
            rows.append({'player_id': f'mock_{player:02d}', 'game_id': f'mock_g_{player}_{game}',
                         'rank': 'A', 'color': color, 'sgf_content': sgf})
    path = ROOT / 'data/mock/train_A.csv'
    write_csv(pd.DataFrame(rows), path)
    print(f'Created {len(rows)} synthetic games: {path}')


if __name__ == '__main__':
    create_mock_data()
