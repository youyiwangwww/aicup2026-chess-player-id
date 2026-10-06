"""Process-wide FINAL TEST 3 input isolation and atomic fail-closed reservation."""
import contextvars
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
_owner = contextvars.ContextVar('test3_owner', default=False)
_stage = contextvars.ContextVar('test3_stage', default='NONE')
_hashing = contextvars.ContextVar('test3_hashing', default=False)


def directory():
    """Return the authoritative ledger, independent of caller configuration."""
    return ROOT/'outputs/phase210'


def require_unconsumed():
    """Existence alone closes the slot; never reopen a receipt to decide eligibility."""
    if any((directory()/name).exists() for name in ['one_shot_attempt.json', 'final_test3_receipt.json', 'CLOSED_TEST', 'FAILED_CLOSED']):
        raise PermissionError('FINAL TEST 3 already reserved/CLOSED; second invocation forbidden')


def _guard(event, args):
    """Protect inputs after reservation, truth until scoring, and immutable registration."""
    if event not in ['open', 'os.remove', 'os.rename']:
        return
    names = args[:2] if event == 'os.rename' else args[:1]
    for name in names:
        if not isinstance(name, (str, bytes, Path)):
            continue
        path = Path(os.fsdecode(name)).resolve()
        protected = path.parent == directory()/'splits' or path.name in ['preregistered_protocol.json', 'preregistered_protocol.sha256'] and path.parent == directory()
        if not protected:
            continue
        mutation = event != 'open' or any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0) & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        registered = (directory()/'preregistered_protocol.json').exists()
        if event == 'os.rename' and not registered and _stage.get() == 'PREPARATION' and not Path(os.fsdecode(args[1])).exists():
            continue
        if mutation and (registered or path.exists()):
            # The sidecar is created exactly once after the manifest itself.
            if event == 'open' and path.name == 'preregistered_protocol.sha256' and not path.exists() and 'x' in (args[1] or ''):
                continue
            raise PermissionError('Preregistered FINAL TEST 3 inputs/protocol are immutable')
        if event == 'open' and not mutation and path.parent == directory()/'splits':
            if not _owner.get():
                require_unconsumed()
            if path.name == 'ground_truth.csv' and _stage.get() != 'SCORING':
                # Hashing bytes is allowed during PREPARATION/PREFLIGHT; CSV reads are not.
                if _stage.get() not in ['PREPARATION', 'PREFLIGHT'] or not _hashing.get():
                    raise PermissionError('FINAL TEST 3 truth may be parsed only after all predictions are saved')


sys.addaudithook(_guard)


def input_digest(path):
    """Permit only this byte-hashing operation before truth is available for scoring."""
    import hashlib
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    token = _hashing.set(True)
    try:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()
    finally:
        _hashing.reset(token)


@contextmanager
def stage(value):
    """Bound truth access to explicit preparation/hash-only or post-inference scoring."""
    token = _stage.set(value)
    try:
        yield
    finally:
        _stage.reset(token)


@contextmanager
def claim_once():
    """Consume the slot atomically before any preprocessing/inference, including failures."""
    require_unconsumed()
    directory().mkdir(parents=True, exist_ok=True)
    with (directory()/'one_shot_attempt.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'RESERVED', 'evaluation_count': 0,
                   'timestamp': datetime.now(timezone.utc).isoformat()}, stream)
    token = _owner.set(True)
    try:
        yield
    except BaseException:
        with (directory()/'FAILED_CLOSED').open('x', encoding='utf-8') as stream:
            stream.write('Attempt failed; inference and automatic retry permanently forbidden.\n')
        raise
    finally:
        _owner.reset(token)
