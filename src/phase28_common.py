"""DEV2 guards, complete score matrices, normalization and equivalent pooling."""
import contextvars
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluate import evaluate
from .player_model import PlayerEncoder

_dev = contextvars.ContextVar('phase28_dev_only', default=False)


def _guard(event, args):
    """Reject all closed TEST input/inference paths while working on DEV2."""
    if event == 'open' and _dev.get() and isinstance(args[0], (str, bytes, Path)):
        path = '/' + str(args[0]).replace('\\', '/').lower().lstrip('/')
        if ('/outputs/splits/test_' in path or '/outputs/round1_archive/' in path
                or '/outputs/phase26/final_test2/' in path or 'final_test2_ground_truth' in path
                or '/outputs/phase26/splits/final_test2_' in path):
            raise PermissionError('Phase 2.8 DEV2 must not open CLOSED TEST paths')


sys.addaudithook(_guard)


@contextmanager
def dev2_only():
    """Install a scoped audit guard around every DEV2 experiment."""
    token = _dev.set(True)
    try:
        yield
    finally:
        _dev.reset(token)


class DeterministicAdaptiveAverage(nn.Module):
    """Identical adaptive-average bins, using deterministic mean/slice autograd."""
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        """Average floor/ceil windows; retain the original parameter-free operator."""
        h, w = x.shape[-2:]
        rows = []
        for i in range(self.size):
            cells = []
            for j in range(self.size):
                cell = x[..., math.floor(i*h/self.size):math.ceil((i+1)*h/self.size),
                         math.floor(j*w/self.size):math.ceil((j+1)*w/self.size)]
                cells.append(cell.mean(dim=(-2, -1)))
            rows.append(torch.stack(cells, dim=-1))
        return torch.stack(rows, dim=-2)


def encoder(config, device):
    """Keep the existing encoder weights/architecture and use equivalent pooling."""
    model = PlayerEncoder(**config['model'])
    model.pool = DeterministicAdaptiveAverage(config['model']['pool_size'])
    return model.to(device)


def rankings(scores, players, questions):
    """Rank every candidate with stable ID tie-breaking and unique Top-5."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(questions), len(players)) or not np.isfinite(scores).all():
        raise ValueError('Invalid complete score matrix')
    if len(set(players)) != len(players) or list(players) != sorted(players):
        raise ValueError('Candidate IDs must be unique and sorted')
    rows = []
    for question, values in zip(questions, scores):
        for rank, index in enumerate(np.argsort(-values, kind='stable')[:5], 1):
            rows.append({'question_id': question, 'rank': rank, 'player_id': players[index], 'similarity': values[index]})
    return pd.DataFrame(rows, columns=['question_id', 'rank', 'player_id', 'similarity'])


def metrics(scores, players, questions, truth):
    """Score only the given DEV2 truths, keeping the entire query denominator."""
    values, details = evaluate(rankings(scores, players, questions), truth)
    return {name: float(values[key]) for name, key in [('top1', 'top_1_accuracy'), ('top3', 'top_3_accuracy'),
             ('top5', 'top_5_accuracy'), ('competition_score', 'competition_score')]}, details


def normalize_scores(scores, method='z-score'):
    """Normalize across candidates separately for each question; flat rows become zero."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite similarities')
    if method == 'z-score':
        offset, scale = scores.mean(axis=1, keepdims=True), scores.std(axis=1, keepdims=True)
    elif method == 'min-max':
        offset = scores.min(axis=1, keepdims=True)
        scale = scores.max(axis=1, keepdims=True) - offset
    else:
        raise ValueError('Normalization must be z-score or min-max')
    return (scores - offset) / np.where(scale == 0, 1, scale)


def fuse(opening, triplet, alpha, normalization='z-score'):
    """Fuse complete normalized scores, preserving both endpoint rankings."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha outside [0,1]')
    return alpha * normalize_scores(opening, normalization) + (1-alpha) * normalize_scores(triplet, normalization)
