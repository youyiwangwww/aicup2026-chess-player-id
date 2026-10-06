"""Export reproducibility metadata without personal or sensitive system paths."""
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd
import torch

from .experiment_state import (experiment_parser, read_json, require_final_test,
                               verify_test_result)
from .metric_utils import file_digest, write_json
from .utils import ROOT, load_config, resolve_path


def sanitized_config(config):
    """Replace absolute paths outside the project and never emit a user/home path."""
    def sanitize(value):
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
                                       or PurePosixPath(value).is_absolute()):
            if not Path(value).is_absolute():
                return '<external-path>'
            try:
                return str(Path(value).resolve().relative_to(ROOT)).replace('\\', '/')
            except ValueError:
                return '<external-path>'
        return value
    return sanitize(config)


def write_experiment_summary(config, total_elapsed_seconds=None):
    """Record the fixed validation choice and final test results for this exact dataset."""
    state, receipt = require_final_test(config)
    metrics = verify_test_result(config, 'triplet', state)
    audit = read_json(config['paths']['split_audit_json'])
    source_sha = file_digest(config['paths']['training_csv'])
    if source_sha != audit['source_csv_sha256']:
        raise ValueError('Source training CSV changed since split construction')
    comparison = pd.read_csv(resolve_path(config['paths']['comparison_csv']))
    if comparison.method.tolist() != ['random', 'opening', 'triplet']:
        raise ValueError('Complete all three test baselines before experiment summary')
    for method in comparison.method:
        expected = verify_test_result(config, method, state)
        row = comparison[comparison.method == method].iloc[0]
        for column, key in [('top1', 'top_1_accuracy'), ('top3', 'top_3_accuracy'),
                            ('top5', 'top_5_accuracy'), ('competition_score', 'competition_score')]:
            if abs(float(row[column]) - float(expected[key])) > 1e-12:
                raise ValueError('Comparison CSV does not match verified test results')
    summary = {'schema_version': 25, 'dataset_kind': config['dataset_kind'], 'seed': config['seed'],
               'python_version': platform.python_version(), 'pytorch_version': str(torch.__version__),
               'cuda_available': torch.cuda.is_available(),
               'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               'config': sanitized_config(config), 'dataset_sha256': source_sha,
               'best_epoch': state['best_epoch'], 'best_validation_score': state['best_validation_score'],
               'final_test_metrics': metrics, 'final_test_evaluations': 1,
               'baselines': comparison.to_dict(orient='records'),
               'training_elapsed_seconds': state['training_elapsed_seconds'],
               'total_elapsed_seconds': total_elapsed_seconds,
               'selected_checkpoint_sha256': receipt['best_checkpoint_sha256'],
               'interpretation': ('Mock data is only pipeline validation.' if config['dataset_kind'] == 'mock'
                                  else 'Final test was not used for training or checkpoint selection.')}
    write_json(summary, config['paths']['experiment_summary_json'])
    print(f'Experiment summary saved; dataset_kind={config["dataset_kind"]}', flush=True)
    return summary


def main():
    """Export metadata for a completed experiment."""
    args = experiment_parser(__doc__).parse_args()
    write_experiment_summary(load_config(args.config))


if __name__ == '__main__':
    main()
