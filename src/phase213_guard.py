"""Protect all Phase 2.12 artifacts while retaining historical TEST isolation."""
from contextlib import contextmanager
import contextvars
import os
from pathlib import Path
import sys
from .phase212_guard import dev3_only

_active=contextvars.ContextVar('phase213_only',default=False)


def _guard(event,args):
    """Deny writes/deletions to the reference experiment and official data."""
    if not _active.get() or event not in ('open','os.remove','os.rename'):
        return
    for name in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(name,(str,bytes,Path)):
            continue
        path=Path(os.fsdecode(name)).resolve().as_posix().lower()
        if 'final_test4' in path or 'final_test_4' in path:
            raise PermissionError('No FINAL TEST 4 in Phase 2.13')
        mutation=event!='open' or any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if mutation and any(fragment in path for fragment in ['/outputs/phase213/splits/','/outputs/phase213/features/']):
            raise PermissionError('No new identities, split or feature preprocessing in Phase 2.13')
        if mutation and ('/outputs/phase212/' in path or path.endswith(('/configs/phase212.yaml','/configs/phase212_best_dev.yaml'))):
            raise PermissionError('Phase 2.13 must preserve Phase 2.12 reference artifacts')


sys.addaudithook(_guard)


@contextmanager
def research_only():
    """Limit all training/diagnostics/reports to frozen DEV3, never CLOSED TEST."""
    with dev3_only():
        token=_active.set(True)
        try:
            yield
        finally:
            _active.reset(token)
