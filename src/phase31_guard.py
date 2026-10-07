"""TRAIN-only policy smoke boundaries; never a Player-ID evaluation entry point."""
from contextlib import contextmanager
import contextvars
import os
from pathlib import Path
import sys
from .phase213_guard import research_only

_active=contextvars.ContextVar('phase31_train_only',default=False)

def require_smoke_operation(operation):
    """Reject retrieval/evaluation for every backend, including bootstrap models."""
    if operation not in {'source_audit','structural_forward','move_statistics','feature_trace','bootstrap','aggregation_smoke'}:
        raise PermissionError('Phase 3.1 forbids Player-ID evaluation and FINAL TEST')

def _audit(event,args):
    """Protect saved experiments, deny VAL reads and any FINAL TEST creation."""
    if not _active.get() or event not in ('open','os.mkdir','os.remove','os.rename'):
        return
    for name in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(name,(str,bytes,Path)):continue
        path=Path(os.fsdecode(name)).resolve().as_posix().lower()
        if 'final_test' in path or '/outputs/phase212/splits/val_' in path:
            raise PermissionError('Policy smoke cannot access VAL or FINAL TEST')
        mutation=event!='open' or any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if mutation and any('/outputs/'+p+'/' in path for p in ['phase212','phase213','phase30']):
            raise PermissionError('Saved research artifacts are immutable')
        if mutation and '/outputs/phase31/' in path and any(s in Path(path).name for s in ['competition','retrieval','player_score','fusion']):
            raise PermissionError('No Player-ID performance artifacts')

sys.addaudithook(_audit)

@contextmanager
def train_smoke_only():
    """Combine historical CLOSED TEST protection with narrower TRAIN-only restrictions."""
    with research_only():
        token=_active.set(True)
        try:yield
        finally:_active.reset(token)
