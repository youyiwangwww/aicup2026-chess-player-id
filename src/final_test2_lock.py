"""Fail-closed, process-wide protection for the consumed FINAL TEST 2 inputs."""
import contextvars
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_owner = contextvars.ContextVar('final_test2_owner', default=False)


def final_directory():
    """Return the single authoritative evaluation ledger directory."""
    return ROOT / 'outputs/phase26/final_test2'


def require_unconsumed(directory=None):
    """Reject completed, running, failed, or previously reserved evaluations."""
    directory = Path(directory or final_directory())
    if any((directory / name).exists() for name in
           ['one_shot_attempt.json', 'final_test2_receipt.json', 'CLOSED_TEST']):
        raise ValueError('FINAL TEST 2 is CLOSED or already reserved; use saved results. Re-evaluation is forbidden.')
    summary = ROOT / 'outputs/phase26/summary.json'
    if summary.exists() and json.loads(summary.read_text(encoding='utf-8')).get('final_test2_evaluations', 0) != 0:
        raise ValueError('FINAL TEST 2 is CLOSED in the experiment summary; re-evaluation is forbidden.')


def _protected(path):
    """Identify fixed inputs/weights/configuration without changing their permissions."""
    return (path in {ROOT / 'configs/phase26_selected.yaml', ROOT / 'data/training/train_A.csv',
                     ROOT / 'outputs/phase26/selected/best.pt',
                     ROOT / 'outputs/phase26/experiments/H20-Hard/train.csv'}
            or path.parent == ROOT / 'outputs/phase26/splits')


def _reject_mutation(path):
    """Keep fixed artifacts immutable after the one-shot slot is reserved."""
    if _protected(path):
        try:
            require_unconsumed()
        except ValueError as exc:
            raise PermissionError(f'Frozen FINAL TEST 2 artifact cannot be modified: {path}') from exc


def _input_guard(event, args):
    """Block actual input opens by every CLI importing the shared utilities."""
    if event in ('os.rename', 'os.remove'):
        for name in args[:2] if event == 'os.rename' else args[:1]:
            if isinstance(name, (str, bytes, Path)):
                _reject_mutation(Path(os.fsdecode(name)).resolve())
        return
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
        return
    name = args[0].decode() if isinstance(args[0], bytes) else str(args[0])
    path = Path(name).resolve()
    mode = args[1] or ''
    flags = args[2] or 0
    if any(value in mode for value in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        _reject_mutation(path)
    prefix = ROOT / 'outputs/phase26/splits'
    if not _owner.get() and path.parent == prefix and path.name.startswith('final_test2_'):
        require_unconsumed()


sys.addaudithook(_input_guard)


@contextmanager
def claim_once(directory=None):
    """Reserve atomically before preprocessing; only this invocation may read inputs."""
    directory = Path(directory or final_directory())
    require_unconsumed(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'one_shot_attempt.json'
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'RESERVED', 'timestamp': timestamp, 'evaluation_count': 0}, stream)
    token = _owner.set(True)
    try:
        yield timestamp
    except BaseException:
        path.write_text(json.dumps({'status': 'FAILED_CLOSED', 'timestamp': timestamp,
                                    'automatic_retry_forbidden': True}), encoding='utf-8')
        raise
    finally:
        _owner.reset(token)
