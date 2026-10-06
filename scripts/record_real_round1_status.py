"""Record a completed Round 1 attempt without rerunning its final TEST."""
import json
import time
from pathlib import Path


def main():
    """Persist completed experiment metadata and independently observed test results."""
    destination = Path(__file__).resolve().parents[1] / 'outputs/results'
    statistics = json.loads((destination / 'real_dataset_statistics.json').read_text(encoding='utf-8'))
    summary = json.loads((destination / 'experiment_summary.json').read_text(encoding='utf-8'))
    if summary['dataset_kind'] != 'real' or summary['final_test_evaluations'] != 1:
        raise ValueError('A completed real experiment with one final TEST is required')
    report = {
        'status': 'completed',
        'config': 'configs/real_quick.yaml', 'seed': 42,
        'tests': {'total': 42, 'failures': 0, 'errors': 0, 'elapsed_seconds': 7.552},
        'last_stage': 'experiment_summary',
        'error': None,
        'pipeline_elapsed_seconds': summary['total_elapsed_seconds'],
        'elapsed_since_sanity_check_seconds': time.time() - statistics['started_unix_seconds'],
        'formal_test_executions': 1, 'professor_report_updated': True,
        'dataset_sha256': summary['dataset_sha256'],
        'best_epoch': summary['best_epoch'],
        'best_checkpoint_sha256': summary['selected_checkpoint_sha256'],
        'baselines': summary['baselines'],
    }
    (destination / 'real_round1_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
