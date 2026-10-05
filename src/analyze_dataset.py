"""Summarize training data without parsing SGFs."""
import pandas as pd
from .utils import argument_parser, load_config, read_csv, write_csv


def analyze_dataset(frame):
    """Return summary, per-player counts and an exact count histogram."""
    counts = frame.groupby('player_id').size().sort_index()
    colors = frame.color.str.strip().str.upper()
    summary = {'total_games': len(frame), 'num_players': len(counts),
               'mean_games_per_player': counts.mean(),
               'median_games_per_player': counts.median(),
               'max_games_per_player': counts.max(),
               'min_games_per_player': counts.min(),
               'black_ratio': colors.eq('B').mean(),
               'white_ratio': colors.eq('W').mean(),
               'unknown_color_ratio': (~colors.isin(['B', 'W'])).mean()}
    rows = [{'section': 'summary', 'key': k, 'value': v} for k, v in summary.items()]
    rows += [{'section': 'player_counts', 'key': k, 'value': v} for k, v in counts.items()]
    rows += [{'section': 'distribution', 'key': str(k), 'value': v}
             for k, v in counts.value_counts().sort_index().items()]
    return pd.DataFrame(rows)


def main():
    """Print statistics and persist them in a single section-tagged CSV."""
    parser = argument_parser(__doc__)
    parser.add_argument('--input', help='Override training CSV path')
    args = parser.parse_args()
    config = load_config(args.config)
    frame = read_csv(args.input or config['paths']['training_csv'],
                     ['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
    result = analyze_dataset(frame)
    print(result.to_string(index=False))
    write_csv(result, config['paths']['statistics_csv'])


if __name__ == '__main__':
    main()
