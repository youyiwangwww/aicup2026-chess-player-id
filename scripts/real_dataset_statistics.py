"""Read-only sanity check of the official training CSV for Round 1."""
import json
import time
from pathlib import Path

import pandas as pd


def main():
    """Validate required fields and save counts without changing the source CSV."""
    root = Path(__file__).resolve().parents[1]
    started = time.time()
    source = root / 'data/training/train_A.csv'
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    counts = frame.groupby('player_id').size()
    colors = frame.color.str.strip().str.upper()
    result = {
        'started_unix_seconds': started,
        'source': 'data/training/train_A.csv',
        'columns': frame.columns.tolist(),
        'total_games': len(frame), 'unique_players': len(counts),
        'games_per_player': {'mean': float(counts.mean()), 'median': float(counts.median()),
                             'min': int(counts.min()), 'max': int(counts.max())},
        'black_games': int(colors.eq('B').sum()), 'white_games': int(colors.eq('W').sum()),
        'black_ratio': float(colors.eq('B').mean()), 'white_ratio': float(colors.eq('W').mean()),
        'unknown_color_games': int((~colors.isin(['B', 'W'])).sum()),
        'duplicate_game_id_count': int(frame.game_id.duplicated().sum()),
        'duplicate_sgf_count': int(frame.sgf_content.duplicated().sum()),
        'duplicate_count_definition': 'Occurrences after the first identical value',
        'players_with_at_least_games': {str(n): int(counts.ge(n).sum()) for n in [10, 15, 20, 50, 100]},
        'empty_required_fields': {key: int(frame[key].str.strip().eq('').sum()) for key in required},
        'color_counts': colors.value_counts().to_dict(),
    }
    destination = root / 'outputs/results/real_dataset_statistics.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(result['empty_required_fields'].values()) or result['unknown_color_games']:
        raise ValueError('Invalid required fields or colors; stop before official experiment')


if __name__ == '__main__':
    main()
