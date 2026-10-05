"""Compute local Top-k accuracy and exponential competition score."""
import math
import pandas as pd
from .utils import argument_parser, load_config, read_csv, resolve_path, write_csv


def evaluate(predictions, truth):
    """Evaluate every ground-truth question, including missing predictions."""
    if truth.empty or truth.question_id.duplicated().any():
        raise ValueError('Ground truth must contain one row per question')
    predictions = predictions.copy()
    predictions['rank'] = pd.to_numeric(predictions['rank'], errors='raise')
    if not predictions['rank'].isin([1, 2, 3, 4, 5]).all():
        raise ValueError('Prediction ranks must be integers in 1..5')
    if (predictions.duplicated(['question_id', 'rank']).any()
            or predictions.duplicated(['question_id', 'player_id']).any()):
        raise ValueError('Duplicate rank or player within a question')
    if not set(predictions.question_id) <= set(truth.question_id):
        raise ValueError('Predictions contain unknown question IDs')
    for _, group in predictions.groupby('question_id'):
        if sorted(group['rank']) != list(range(1, len(group) + 1)):
            raise ValueError('Ranks must start at 1 and be contiguous')
    rows = []
    for row in truth.itertuples(index=False):
        match = predictions[(predictions.question_id == row.question_id)
                            & (predictions.player_id == row.player_id)]
        rank = int(match.iloc[0]['rank']) if len(match) else 0
        rows.append({'question_id': row.question_id, 'player_id': row.player_id,
                     'correct_rank': rank, 'top_1': int(rank == 1),
                     'top_3': int(1 <= rank <= 3), 'top_5': int(1 <= rank <= 5),
                     'competition_score': math.exp(-(rank - 1)) if rank else 0.0})
    details = pd.DataFrame(rows)
    metrics = {name: float(details[column].mean()) for name, column in
               [('top_1_accuracy', 'top_1'), ('top_3_accuracy', 'top_3'),
                ('top_5_accuracy', 'top_5'), ('competition_score', 'competition_score')]}
    return metrics, details


def main():
    """Print aggregate metrics and save aggregate and per-question results."""
    args = argument_parser(__doc__).parse_args()
    paths = load_config(args.config)['paths']
    truth = read_csv(paths['ground_truth_csv'], ['question_id', 'player_id'])
    # A header-only prediction file is valid when all query SGFs fail.
    predictions = pd.read_csv(resolve_path(paths['predictions_csv']), dtype=str, keep_default_na=False)
    required = {'question_id', 'rank', 'player_id'}
    if not required <= set(predictions.columns):
        raise ValueError(f'Missing prediction columns: {required - set(predictions.columns)}')
    metrics, details = evaluate(predictions, truth)
    for name, value in metrics.items():
        print(f'{name}: {value:.6f}')
    write_csv(pd.DataFrame([metrics]), paths['metrics_csv'])
    write_csv(details, paths['evaluation_csv'])


if __name__ == '__main__':
    main()
