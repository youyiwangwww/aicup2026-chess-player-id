"""Small helpers shared by Phase 2, without importing torch at module load."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from .utils import ROOT, argument_parser, resolve_path


def metric_parser(description):
    """Use the quick config by default for Phase 2 commands."""
    parser = argument_parser(description)
    parser.set_defaults(config=str(ROOT / 'configs/triplet_quick.yaml'))
    return parser


def file_digest(path):
    """Hash CSV contents to detect stale splits, caches and checkpoints."""
    digest = hashlib.sha256()
    with resolve_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value, path):
    """Atomically commit JSON only after all work succeeds."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(stream.name)
    temporary.replace(path)


def seed_everything(seed, deterministic=True, num_threads=4):
    """Seed Python, numpy and torch; deterministic runs assume the same environment."""
    import torch

    if seed < 0 or num_threads < 1:
        raise ValueError('seed must be nonnegative and num_threads must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def choose_device(name='auto'):
    """Select CUDA when available, with an explicit CPU option for smoke tests."""
    import torch

    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in ('cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; install a CUDA build or use cpu')
    return torch.device(name)
