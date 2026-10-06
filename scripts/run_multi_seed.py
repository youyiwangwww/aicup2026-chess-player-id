"""Run independent three-way experiments and aggregate per-seed mean/sample std."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys

# Allow the documented `python scripts/run_multi_seed.py` invocation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.run_experiment import MISSING_DATA_MESSAGE
from src.utils import load_config, resolve_path, write_csv


def seed_config(config, seed, output_directory):
    """Move every output path to this seed directory while preserving the source CSV."""
    result = copy.deepcopy(config)
    old_root = resolve_path(config['output_dir'])
    new_root = resolve_path(output_directory)
    result['seed'] = seed
    result['output_dir'] = str(output_directory).replace('\\', '/')
    for key, path in config['paths'].items():
        if key == 'training_csv':
            continue
        try:
            relative = resolve_path(path).relative_to(old_root)
        except ValueError as exc:
            raise ValueError('All experiment output paths must be under output_dir') from exc
        target = new_root / relative
        try:
            target = target.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        result['paths'][key] = str(target).replace('\\', '/')
    return result


def summarize_seeds(tables):
    """Append mean/std rows; sample std uses ddof=1, or 0 for one seed."""
    result = pd.concat(tables, ignore_index=True)
    if result.duplicated(['method', 'seed']).any():
        raise ValueError('Duplicate method/seed results')
    metrics = ['top1', 'top3', 'top5', 'competition_score']
    rows = []
    for method, group in result.groupby('method', sort=False):
        rows.append({'method': method, 'seed': 'mean', **group[metrics].mean().to_dict()})
        rows.append({'method': method, 'seed': 'std',
                     **group[metrics].std(ddof=1 if len(group) > 1 else 0).to_dict()})
    return pd.concat([result, pd.DataFrame(rows)], ignore_index=True)[['method', 'seed', *metrics]]


def main():
    """Launch only when manually requested; the quick script never calls this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/real_quick.yaml')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2026])
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--summary', default='outputs/results/multi_seed_summary.csv')
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique nonnegative integers')
    config = load_config(resolve_path(args.config))
    if not resolve_path(config['paths']['training_csv']).exists():
        parser.exit(1, f'{MISSING_DATA_MESSAGE}\n')
    tables = []
    for seed in args.seeds:
        output = Path(args.output_root) / f'seed_{seed}'
        settings = seed_config(config, seed, output)
        path = resolve_path(output / 'config.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            yaml.safe_dump(settings, stream, allow_unicode=True, sort_keys=False)
        subprocess.run([sys.executable, '-m', 'src.run_experiment', '--config', str(path)],
                        cwd=PROJECT_ROOT, check=True)
        tables.append(pd.read_csv(resolve_path(settings['paths']['comparison_csv'])).assign(seed=seed))
    summary = summarize_seeds(tables)
    write_csv(summary, args.summary)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
