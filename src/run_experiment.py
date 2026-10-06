"""Run the three-way experiment in separate stages, with TEST strictly after training."""
import subprocess
import sys
import time

from .build_metric_split import is_three_way
from .experiment_state import experiment_parser
from .experiment_summary import write_experiment_summary
from .metric_utils import write_json
from .utils import ROOT, load_config, resolve_path

MISSING_DATA_MESSAGE = 'Please place the official train_A.csv at data/training/train_A.csv'
STAGES = ['build_metric_split', 'player_features', 'analyze_feature_coverage', 'train_triplet',
          'evaluate_test', 'random_baseline', 'opening_test', 'compare_baselines']


def run_experiment(config_path, create_mock=False):
    """Fail before any real outputs when the official source CSV is absent."""
    config_path = resolve_path(config_path)
    config = load_config(config_path)
    if not is_three_way(config):
        raise ValueError('Use a Phase 2.5 three-way config')
    source = resolve_path(config['paths']['training_csv'])
    started = time.perf_counter()
    if config['dataset_kind'] == 'real' and create_mock:
        raise ValueError('Mock generation is forbidden for a real-data experiment')
    if create_mock:
        if config['dataset_kind'] != 'mock' or source == ROOT / 'data/training/train_A.csv':
            raise ValueError('Mock generation requires an explicit separate mock source path')
        subprocess.run([sys.executable, '-m', 'src.create_metric_mock_data', '--players', '14',
                        '--output', str(source)], cwd=ROOT, check=True)
    if not source.exists():
        message = MISSING_DATA_MESSAGE if config['dataset_kind'] == 'real' else 'Generate the Phase 2.5 mock CSV first'
        raise FileNotFoundError(message)
    for index, stage in enumerate(STAGES, 1):
        print(f'[{index}/9] {stage}', flush=True)
        subprocess.run([sys.executable, '-m', f'src.{stage}', '--config', str(config_path)],
                        cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print('[9/9] experiment_summary', flush=True)
    summary = write_experiment_summary(config, elapsed)
    elapsed = time.perf_counter() - started
    summary['total_elapsed_seconds'] = elapsed
    write_json(summary, config['paths']['experiment_summary_json'])
    print(f'Completed {config["dataset_kind"]} experiment in {elapsed:.2f}s', flush=True)
    return summary


def main():
    """Run seed 42 by default; multi-seed is a separate user-invoked command."""
    parser = experiment_parser(__doc__)
    parser.add_argument('--create-mock', action='store_true')
    args = parser.parse_args()
    try:
        run_experiment(args.config, args.create_mock)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
