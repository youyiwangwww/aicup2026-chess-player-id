"""Enforce completed-training/one-final-test lifecycle and shared test provenance."""
import json
import hashlib

import pandas as pd

from .build_metric_split import verify_partition_hashes
from .metric_utils import file_digest, metric_parser, write_json
from .utils import ROOT, resolve_path, write_csv


def experiment_parser(description):
    """Default new experiment commands to the real three-way quick config."""
    parser = metric_parser(description)
    parser.set_defaults(config=str(ROOT / 'configs/real_quick.yaml'))
    return parser


def read_json(path):
    """Read local experiment metadata without executing serialized code."""
    with resolve_path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def experiment_signature(config):
    """Bind lifecycle metadata to model/training/split settings, excluding machine paths."""
    settings = {key: config[key] for key in ['seed', 'split', 'features', 'model', 'training', 'inference']}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def require_training_complete(config):
    """Refuse test access before the selected checkpoint is fixed."""
    path = resolve_path(config['paths']['training_state_json'])
    if not path.exists():
        raise ValueError('Training must complete before final test evaluation')
    state = read_json(path)
    if state['status'] != 'completed':
        raise ValueError('Training must complete before final test evaluation')
    if state.get('experiment_signature') != experiment_signature(config):
        raise ValueError('Experiment settings changed after training; restore config or start a new run')
    if state['best_checkpoint_sha256'] != file_digest(config['paths']['best_checkpoint']):
        raise ValueError('Selected checkpoint changed after training completed')
    audit = verify_partition_hashes(config, ['metric_train_csv', 'val_candidates_csv',
                                            'val_queries_csv', 'val_ground_truth_csv'])
    if state['partition_sha256'] != {k: audit['partition_sha256'][k] for k in state['partition_sha256']}:
        raise ValueError('Training state belongs to another split')
    return state


def test_source_hashes(config):
    """Verify the immutable test split against the split-stage audit."""
    keys = ['test_candidates_csv', 'test_queries_csv', 'test_ground_truth_csv']
    verify_partition_hashes(config, keys)
    return {key: file_digest(config['paths'][key]) for key in keys}


def require_final_test(config):
    """Allow other test baselines/comparison only after Triplet's final test is saved."""
    state = require_training_complete(config)
    path = resolve_path(config['paths']['final_test_receipt_json'])
    if not path.exists():
        raise ValueError('Run evaluate_test after completed training before test baselines')
    receipt = read_json(path)
    if receipt['run_id'] != state['run_id'] or receipt['test_source_sha256'] != test_source_hashes(config):
        raise ValueError('Final test result belongs to another training run or test split')
    return state, receipt


def result_paths(config, method):
    """Map each test method to its three result CSVs."""
    paths = config['paths']
    prefix = 'test' if method == 'triplet' else f'{method}_test'
    return {name: paths[f'{prefix}_{name}_csv'] for name in ['predictions', 'metrics', 'evaluation']}


def save_test_result(config, method, predictions, metrics, details, state):
    """Persist fixed test predictions/metrics with source hashes for fair comparison."""
    paths = result_paths(config, method)
    write_csv(predictions, paths['predictions'])
    write_csv(pd.DataFrame([metrics]), paths['metrics'])
    write_csv(details, paths['evaluation'])
    metadata = {'method': method, 'run_id': state['run_id'],
                'best_checkpoint_sha256': state['best_checkpoint_sha256'],
                'test_source_sha256': test_source_hashes(config),
                'result_sha256': {key: file_digest(value) for key, value in paths.items()}}
    write_json(metadata, resolve_path(paths['predictions']).with_suffix('.metadata.json'))
    return metadata


def verify_test_result(config, method, state):
    """Reject edited/stale result files without recomputing or re-selecting test scores."""
    paths = result_paths(config, method)
    metadata = read_json(resolve_path(paths['predictions']).with_suffix('.metadata.json'))
    if (metadata['run_id'] != state['run_id']
            or metadata['test_source_sha256'] != test_source_hashes(config)
            or metadata['best_checkpoint_sha256'] != state['best_checkpoint_sha256']
            or metadata['result_sha256'] != {key: file_digest(value) for key, value in paths.items()}):
        raise ValueError(f'Stale or modified {method} test results')
    return pd.read_csv(resolve_path(paths['metrics'])).iloc[0].to_dict()
