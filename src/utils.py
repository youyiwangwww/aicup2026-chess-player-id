"""Shared configuration, CSV and command-line helpers."""
import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Load YAML; all relative data paths are relative to the project root."""
    with Path(path or ROOT / 'configs/baseline.yaml').open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_path(path):
    """Resolve a configured path independently of the current directory."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path, required, allow_empty=('sgf_content', 'color')):
    """Preserve IDs; leave empty SGF/color values for per-game error handling."""
    frame = pd.read_csv(resolve_path(path), dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('CSV contains no rows')
    for col in required:
        if col not in allow_empty and frame[col].str.strip().eq('').any():
            raise ValueError(f'Empty required value in column: {col}')
    return frame


def write_csv(frame, path):
    """Create the output directory and write a UTF-8 CSV."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def argument_parser(description):
    """Provide the common configuration argument."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default=str(ROOT / 'configs/baseline.yaml'))
    return parser
