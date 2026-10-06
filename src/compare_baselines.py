"""Compare preserved opening fingerprints and Triplet embeddings on identical questions."""
import json

import pandas as pd

from .baseline import predict
from .build_metric_split import is_three_way, read_metric_partitions
from .evaluate import evaluate
from .metric_utils import file_digest, metric_parser
from .opening_features import build_fingerprints
from .utils import load_config, resolve_path, write_csv


def compare(config):
    """Compute both methods with one ground truth and preserve separate output files."""
    if is_three_way(config):
        return compare_test_baselines(config)
    _, candidates, queries, truth = read_metric_partitions(config)
    paths = config['paths']
    prediction_path = resolve_path(paths['predictions_csv'])
    with prediction_path.with_suffix('.metadata.json').open(encoding='utf-8') as stream:
        provenance = json.load(stream)
    if (provenance['prediction_sha256'] != file_digest(prediction_path)
            or provenance['candidate_source_sha256'] != file_digest(paths['candidates_csv'])
            or provenance['query_source_sha256'] != file_digest(paths['queries_csv'])):
        raise ValueError('Stale Triplet predictions; rerun embed_players on this split before comparing')
    cfp, cerr = build_fingerprints(candidates, 'player_id', **config['opening'])
    qfp, qerr = build_fingerprints(queries, 'question_id', **config['opening'])
    write_csv(pd.concat([cerr, qerr], ignore_index=True), paths['opening_errors_csv'])
    opening_predictions = predict(cfp, qfp, config['inference']['top_k'])
    write_csv(opening_predictions, paths['opening_predictions_csv'])
    triplet_predictions = pd.read_csv(resolve_path(paths['predictions_csv']), dtype=str, keep_default_na=False)
    rows = []
    for name, predictions in [('opening', opening_predictions), ('triplet', triplet_predictions)]:
        metrics, details = evaluate(predictions, truth)
        rows.append({'method': name, 'top1': metrics['top_1_accuracy'],
                     'top3': metrics['top_3_accuracy'], 'top5': metrics['top_5_accuracy'],
                     'competition_score': metrics['competition_score']})
        if name == 'opening':
            write_csv(pd.DataFrame([metrics]), paths['opening_metrics_csv'])
            write_csv(details, paths['opening_evaluation_csv'])
    result = pd.DataFrame(rows)
    write_csv(result, paths['comparison_csv'])
    print(result.to_string(index=False), flush=True)
    return result


def compare_test_baselines(config):
    """Compare saved final-test metrics; never run another Triplet test evaluation."""
    from .experiment_state import require_final_test, verify_test_result

    state, _ = require_final_test(config)
    rows = []
    for method in ['random', 'opening', 'triplet']:
        metrics = verify_test_result(config, method, state)
        rows.append({'method': method, 'top1': metrics['top_1_accuracy'],
                     'top3': metrics['top_3_accuracy'], 'top5': metrics['top_5_accuracy'],
                     'competition_score': metrics['competition_score']})
    result = pd.DataFrame(rows)
    write_csv(result, config['paths']['comparison_csv'])
    print('FINAL TEST BASELINE COMPARISON', flush=True)
    print(result.to_string(index=False), flush=True)
    return result


def main():
    """Run opening inference automatically and compare with saved Triplet predictions."""
    args = metric_parser(__doc__).parse_args()
    compare(load_config(args.config))


if __name__ == '__main__':
    main()
