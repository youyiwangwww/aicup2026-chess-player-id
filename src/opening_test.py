"""Run the preserved Opening Fingerprint on the same final held-out test."""
import pandas as pd

from .baseline import predict
from .evaluate import evaluate
from .experiment_state import experiment_parser, require_final_test, save_test_result
from .opening_features import build_fingerprints
from .utils import load_config, read_csv, write_csv


def run_opening_test(config):
    """Use exactly the test candidate/query CSVs already used by Triplet retrieval."""
    state, _ = require_final_test(config)
    candidates = read_csv(config['paths']['test_candidates_csv'],
                          ['player_id', 'game_id', 'color', 'sgf_content'])
    queries = read_csv(config['paths']['test_queries_csv'],
                       ['question_id', 'game_id', 'color', 'sgf_content'])
    truth = read_csv(config['paths']['test_ground_truth_csv'], ['question_id', 'player_id'])
    cfp, cerr = build_fingerprints(candidates, 'player_id', **config['opening'])
    qfp, qerr = build_fingerprints(queries, 'question_id', **config['opening'])
    write_csv(pd.concat([cerr, qerr], ignore_index=True), config['paths']['opening_test_errors_csv'])
    predictions = predict(cfp, qfp, config['inference']['top_k'])
    metrics, details = evaluate(predictions, truth)
    save_test_result(config, 'opening', predictions, metrics, details, state)
    print(f'Opening final-test baseline: {metrics}', flush=True)
    return metrics


def main():
    """Run opening retrieval after the final checkpoint has been fixed."""
    args = experiment_parser(__doc__).parse_args()
    run_opening_test(load_config(args.config))


if __name__ == '__main__':
    main()
