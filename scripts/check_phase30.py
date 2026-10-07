"""Run original tests FIRST, then Phase 3.0 CPU utilities; no experiment execution."""
import hashlib
import compileall
from contextlib import redirect_stdout
import io
import importlib
import json
from pathlib import Path
import py_compile
import pkgutil
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def snapshot():
    """Hash protected research artifacts without opening any CLOSED TEST directory."""
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('outputs/phase212','outputs/phase213')
            for p in (ROOT/folder).rglob('*') if p.is_file()}

def main():
    """Fail immediately on old regressions; record verification without performance results."""
    import torch
    torch.set_num_threads(1)
    before=snapshot()
    suite=unittest.TestSuite()
    for path in sorted((ROOT/'tests').glob('test_*.py')):
        if path.name!='test_phase30.py':
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=path.name))
    # Original tests include tiny temporary synthetic training fixtures, not real experiments.
    with redirect_stdout(io.StringIO()):
        old=unittest.TextTestRunner(verbosity=1).run(suite)
    if not old.wasSuccessful() or old.testsRun!=160:
        raise SystemExit('Original tests failed; stop Phase 3.0')
    new=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_phase30.py'))
    if not new.wasSuccessful() or new.testsRun<14:
        raise SystemExit('Phase 3.0 tests failed')
    modules=[m.name for m in pkgutil.iter_modules([str(ROOT/'src')])]
    for module in modules:
        importlib.import_module('src.'+module)
        py_compile.compile(str(ROOT/'src'/f'{module}.py'),doraise=True)
    if not all(compileall.compile_dir(ROOT/name,quiet=1) for name in ['src','scripts','tests']):
        raise SystemExit('Syntax check failed')
    protocol=json.loads((ROOT/'outputs/phase213/protocol.json').read_text(encoding='utf-8'))
    for mapping in ['phase212_snapshot_sha256','new_code_sha256']:
        for relative,expected in protocol[mapping].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                raise SystemExit('Frozen protocol binding changed: '+relative)
    if snapshot()!=before:
        raise SystemExit('Protected Phase 2 artifacts changed')
    target=ROOT/'outputs/phase30'
    target.mkdir(parents=True,exist_ok=True)
    (target/'verification.json').write_text(json.dumps({
        'original_tests':old.testsRun,'new_tests':new.testsRun,'total_tests':old.testsRun+new.testsRun,
        'passed':True,'imports_and_syntax':'passed','phase212_phase213_artifacts_unchanged':True,
        'frozen_protocol_source_and_artifact_bindings':'passed','imported_module_count':len(modules),
        'real_experiment_training_executed':False,'real_model_inference_executed':False,
        'original_synthetic_unit_fixtures_executed':True,
        'new_split_created':False,'new_performance_generated':False},indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
