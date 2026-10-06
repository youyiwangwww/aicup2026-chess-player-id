"""Compare opening windows on DEV VALIDATION only."""
import argparse

import pandas as pd

from .baseline import predict
from .evaluate import evaluate
from .opening_features import build_fingerprints
from .phase26_common import dev_frames, dev_only
from .utils import ROOT, load_config, write_csv


def opening_metrics(candidates, queries, truth, window, board_size=19):
    """Use all moves for full-game heatmaps, retaining per-game SGF failures."""
    # A finite bound above the SGF text length includes every possible move node.
    bound = window if window is not None else max(
        candidates.sgf_content.str.len().max(), queries.sgf_content.str.len().max()) + 1
    cf, ce = build_fingerprints(candidates, 'player_id', int(bound), board_size)
    qf, qe = build_fingerprints(queries, 'question_id', int(bound), board_size)
    metrics, _ = evaluate(predict(cf, qf, 5), truth)
    return metrics, pd.concat([ce, qe], ignore_index=True)


def analyze_opening(config):
    """Evaluate each requested window on exactly the same DEV questions."""
    with dev_only():
        _, candidates, queries, truth = dev_frames(config)
        rows, errors = [], []
        for window in config['diagnosis']['windows'] + [None]:
            name = f'Opening-{window}' if window is not None else 'Full-game heatmap'
            metrics, failures = opening_metrics(candidates, queries, truth, window)
            rows.append({'experiment': name, 'val_top1': metrics['top_1_accuracy'],
                         'val_top3': metrics['top_3_accuracy'], 'val_top5': metrics['top_5_accuracy'],
                         'val_score': metrics['competition_score'], 'failed_games': len(failures)})
            errors.append(failures.assign(experiment=name))
            print(f'{name}: DEV VAL {metrics}', flush=True)
        result = pd.DataFrame(rows)
        write_csv(result, 'outputs/phase26/opening_signal.csv')
        write_csv(pd.concat(errors, ignore_index=True), 'outputs/phase26/opening_errors.csv')
        return result


def main():
    """Run opening diagnosis with a guarded DEV-only configuration."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/phase26.yaml'))
    analyze_opening(load_config(parser.parse_args().config))


if __name__ == '__main__':
    main()
