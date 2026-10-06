"""Exploratory DEV2-only file scope, preserving all historical experiment artifacts."""
import contextvars
from contextlib import contextmanager
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
_active=contextvars.ContextVar('phase211_dev_only',default=False)


def _guard(event,args):
    """Refuse old TEST inputs/scores, Stability fitting and historical writes."""
    if not _active.get() or event not in ['open','os.remove','os.rename']:
        return
    for name in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(name,(str,bytes,Path)):
            continue
        path=Path(os.fsdecode(name)).resolve().as_posix().lower()
        closed=('/outputs/round1_archive/' in path or '/outputs/splits/test_' in path
                or '/outputs/phase26/final_test2/' in path or '/outputs/phase26/splits/final_test2_' in path
                or '/outputs/phase210/' in path
                or any('/outputs/results/'+prefix in path for prefix in ['test_','random_test_','opening_test_','final_test_']))
        if closed:
            raise PermissionError('Phase 2.11 refuses every CLOSED TEST input, truth and score matrix')
        if '/outputs/phase29/' in path and not path.endswith('/summary.json'):
            raise PermissionError('Stability data/truth cannot be used for DEV forensics')
        if '/outputs/phase28/experiments/' in path and not path.endswith('/triplet-hard-100/best.pt'):
            raise PermissionError('Phase 2.11 cannot read/select another checkpoint')
        mutation=event!='open' or any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if mutation and ('/outputs/phase28/' in path or path.endswith('/configs/phase28_selected.yaml') or '/data/' in path):
            raise PermissionError('Historical data/config/checkpoints are immutable')


sys.addaudithook(_guard)


@contextmanager
def dev_only():
    """Wrap every Phase 2.11 entry point, including reporting, in the same file policy."""
    token=_active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


def closed_markers():
    """Confirm consumed historical slots by marker existence without opening their files."""
    markers={'round1':'outputs/results/final_test_receipt.json',
             'final_test2':'outputs/phase26/final_test2/CLOSED_TEST',
             'final_test3':'outputs/phase210/CLOSED_TEST'}
    statuses={key:(ROOT/path).exists() for key,path in markers.items()}
    if not all(statuses.values()):
        raise ValueError('Missing historical CLOSED/consumed marker; refusing forensics')
    return statuses
