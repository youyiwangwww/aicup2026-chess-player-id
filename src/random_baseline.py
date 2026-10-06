"""A reproducible random ranking on the exact final-test candidate pool."""
import hashlib

import numpy as np
import pandas as pd

from .evaluate import evaluate
from .experiment_state import experiment_parser, require_final_test, save_test_result
from .utils import load_config, read_csv


def random_predictions(players, questions, seed=42, top_k=5):
    """Permute unique sorted candidates using a deterministic per-question RNG."""
    players = sorted(set(players))
    if not players or seed < 0 or not 1 <= top_k <= 5:
        raise ValueError('Need candidates, seed>=0 and top_k in 1..5')
    rows = []
    for question in sorted(set(questions)):
        material = f'{seed}\0random-ranking\0{question}'.encode()
        question_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], 'little')
        rng = np.random.default_rng(question_seed)
        for rank, index in enumerate(rng.permutation(len(players))[:top_k], 1):
            rows.append({'question_id': question, 'rank': rank,
                         'player_id': players[index], 'similarity': 0.0})
    return pd.DataFrame(rows, columns=['question_id', 'rank', 'player_id', 'similarity'])


def run_random_baseline(config):
    """Score random predictions after final test is fixed, using all original candidates."""
    state, _ = require_final_test(config)
    candidates = read_csv(config['paths']['test_candidates_csv'], ['player_id', 'game_id'])
    queries = read_csv(config['paths']['test_queries_csv'], ['question_id', 'game_id'])
    truth = read_csv(config['paths']['test_ground_truth_csv'], ['question_id', 'player_id'])
    predictions = random_predictions(candidates.player_id, queries.question_id, config['seed'],
                                     config['inference']['top_k'])
    metrics, details = evaluate(predictions, truth)
    save_test_result(config, 'random', predictions, metrics, details, state)
    print(f'Random final-test baseline: {metrics}', flush=True)
    return metrics


def main():
    """Run the fixed-seed baseline on a completed final test."""
    args = experiment_parser(__doc__).parse_args()
    run_random_baseline(load_config(args.config))


if __name__ == '__main__':
    main()
