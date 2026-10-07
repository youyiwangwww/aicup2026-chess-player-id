"""Original191 tests FIRST, then policy structural tests; no research experiments."""
import compileall
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import importlib
import json
from pathlib import Path
import pkgutil
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def snapshot():
    """Read protected DEV artifacts only, excluding all historical CLOSED TEST folders."""
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ['phase212','phase213','phase30'] for p in (ROOT/'outputs'/name).rglob('*') if p.is_file()}

def main():
    """Capture synthetic-fixture metrics in test log, not as new method performance."""
    started=time.perf_counter();before=snapshot()
    out=ROOT/'outputs/phase31';out.mkdir(parents=True,exist_ok=True)
    with (out/'tests.log').open('w',encoding='utf-8') as stream,redirect_stdout(stream),redirect_stderr(stream):
        suite=unittest.TestSuite()
        for p in sorted((ROOT/'tests').glob('test_*.py')):
            if p.name not in {'test_phase31.py','test_phase31b.py'}:suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p.name))
        old=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        if not old.wasSuccessful() or old.testsRun!=191:raise SystemExit('Original191tests failed')
        new=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_phase31.py'))
    if not new.wasSuccessful() or new.testsRun<20:raise SystemExit('Phase31tests failed; see tests.log')
    if not all(compileall.compile_dir(ROOT/f,quiet=1) for f in ['src','scripts','tests']):raise SystemExit('Syntax failure')
    for m in pkgutil.iter_modules([str(ROOT/'src')]):importlib.import_module('src.'+m.name)
    if before!=snapshot():raise SystemExit('Protected artifacts changed')
    protocol=json.loads((ROOT/'outputs/phase213/protocol.json').read_text(encoding='utf-8'))
    for field in ['phase212_snapshot_sha256','new_code_sha256']:
        for path,digest in protocol[field].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise SystemExit('Frozen binding changed')
    report={'original_tests':old.testsRun,'new_tests':new.testsRun,'total_tests':old.testsRun+new.testsRun,
        'passed':True,'imports_passed':True,'syntax_passed':True,'protected_artifacts_unchanged':True,
        'frozen_bindings_verified':True,'elapsed_seconds':time.perf_counter()-started,
        'real_player_id_evaluation':False,'original_synthetic_fixtures_executed':True}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)

if __name__=='__main__':main()
