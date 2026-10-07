"""CPU preparation only: no training commands, TEST inputs, new splits or performance outputs."""
import contextvars
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from .phase213_guard import research_only

_active=contextvars.ContextVar('phase30_cpu_only',default=False)
ALLOWED_ACTIONS={'static_analysis','adapter','mapping','aggregation','tests','documentation'}


def require_cpu_preparation(action):
    """Reject model-experiment APIs explicitly, instead of allowing a silent CPU fallback."""
    if action not in ALLOWED_ACTIONS:
        raise PermissionError('Phase 3.0 permits only static audit and CPU preparation')


def _audit(event,args):
    """Block process launches and protected artifacts while running Phase 3.0 utilities."""
    if not _active.get():
        return
    if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn'):
        raise PermissionError('Phase 3.0 cannot launch training/build/inference subprocesses')
    if event not in ('open','os.remove','os.rename','os.mkdir'):
        return
    for name in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(name,(str,bytes,Path)):
            continue
        path=Path(os.fsdecode(name)).resolve().as_posix().lower()
        if 'final_test' in path or '/outputs/phase30/splits' in path:
            raise PermissionError('Phase 3.0 cannot create/open FINAL TEST or new split')
        mutation=event!='open' or any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if mutation and '/outputs/phase213/' in path:
            raise PermissionError('Phase 2.13 artifacts are read-only')
        if mutation and '/outputs/phase30/' in path and any(s in Path(path).name for s in ['prediction','performance','score','result']):
            raise PermissionError('Phase 3.0 cannot produce performance results')


sys.addaudithook(_audit)


@contextmanager
def cpu_preparation_only():
    """Combine CPU-only execution rules with existing CLOSED TEST/reference protections."""
    with research_only():
        token=_active.set(True)
        try:
            yield
        finally:
            _active.reset(token)
