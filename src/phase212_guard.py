"""DEV3-only scope; historical datasets/checkpoints/scores are unavailable for selection."""
import contextvars
from contextlib import contextmanager
import os
from pathlib import Path
import sys

_active=contextvars.ContextVar('phase212_dev3_only',default=False)


def _guard(event,args):
    """Permit only copied identity metadata from Phase 2.10, never historical model/data reads."""
    if not _active.get() or event not in ['open','os.remove','os.rename']:
        return
    for name in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(name,(str,bytes,Path)):
            continue
        path=Path(os.fsdecode(name)).resolve().as_posix().lower()
        allowed_meta=path.endswith(('/outputs/phase210/eligibility.json','/outputs/phase210/inference_complete.json'))
        historical=('/outputs/round1_archive/' in path or '/outputs/splits/' in path or '/outputs/results/' in path
                    or any('/outputs/'+phase+'/' in path for phase in ['phase26','phase28','phase29','phase210','phase211']))
        if historical and not allowed_meta:
            raise PermissionError('Phase 2.12 refuses historical TEST/DEV/Stability data, scores and checkpoints')
        mutation=event!='open' or any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if mutation and path.endswith(('/outputs/phase212/eligibility.json','/outputs/phase212/blocked_summary.json')):
            raise PermissionError('Original blocked Phase 2.12 record must be preserved')
        if mutation and '/outputs/phase212/splits/' in path:
            audit=Path(path).parent/'split_audit.json'
            if audit.exists():
                raise PermissionError('DEV3 split is frozen')
        if mutation and path.endswith(('/outputs/phase212/revised_protocol.json','/outputs/phase212/train_diagnostic_subset.json')) and Path(path).exists():
            raise PermissionError('User revision and fixed TRAIN diagnostic subset are immutable')
        if mutation and (historical or '/data/' in path or path.endswith('/configs/phase28_selected.yaml')):
            raise PermissionError('Historical data and identity metadata are immutable')


sys.addaudithook(_guard)


@contextmanager
def dev3_only():
    """Apply the same historical isolation to eligibility, training and report entry points."""
    token=_active.set(True)
    try:
        yield
    finally:
        _active.reset(token)
