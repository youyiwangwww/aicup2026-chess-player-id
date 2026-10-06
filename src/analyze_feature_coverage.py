"""Report preprocessing coverage separately for each train/val/test partition."""
import numpy as np
import pandas as pd

from .experiment_state import experiment_parser, read_json
from .metric_utils import file_digest
from .player_features import FEATURE_VERSION
from .utils import load_config, write_csv

PARTITION_MANIFESTS = [
    ('train', 'metric_train_csv', 'train_manifest'),
    ('validation_candidate', 'val_candidates_csv', 'val_candidate_manifest'),
    ('validation_query', 'val_queries_csv', 'val_query_manifest'),
    ('test_candidate', 'test_candidates_csv', 'test_candidate_manifest'),
    ('test_query', 'test_queries_csv', 'test_query_manifest'),
]


def coverage_row(name, manifest):
    """Count failed games as zero sampled positions in all per-game statistics."""
    total = manifest['input_games']
    valid = manifest['valid_games']
    failed = manifest['skipped_games']
    if total != valid + failed or valid != len(manifest['records']) or total < 1:
        raise ValueError('Inconsistent or empty feature manifest')
    positions = np.array([record['num_positions'] for record in manifest['records']] + [0] * failed)
    return {'partition': name, 'total_games': total, 'successful_games': valid,
            'failed_games': failed, 'success_rate': valid / total,
            'mean_sampled_positions': float(positions.mean()),
            'median_sampled_positions': float(np.median(positions)),
            'min_sampled_positions': int(positions.min()),
            'max_sampled_positions': int(positions.max()),
            'total_sampled_positions': int(positions.sum())}


def analyze_coverage(config):
    """Verify cache sources and summarize all five partitions without loading tensors."""
    rows = []
    for name, csv_key, manifest_key in PARTITION_MANIFESTS:
        manifest = read_json(config['paths'][manifest_key])
        if (manifest['source_sha256'] != file_digest(config['paths'][csv_key])
                or manifest['seed'] != config['seed'] or manifest['features'] != config['features']
                or manifest['feature_version'] != FEATURE_VERSION):
            raise ValueError(f'Stale {name} feature manifest; rerun preprocessing')
        rows.append(coverage_row(name, manifest))
    result = pd.DataFrame(rows)
    write_csv(result, config['paths']['feature_coverage_csv'])
    print(result.to_string(index=False), flush=True)
    return result


def main():
    """Write feature coverage for a preprocessed three-way split."""
    args = experiment_parser(__doc__).parse_args()
    analyze_coverage(load_config(args.config))


if __name__ == '__main__':
    main()
