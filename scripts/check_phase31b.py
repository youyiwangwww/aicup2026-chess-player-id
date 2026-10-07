"""Preserve the original 216 tests and distinguish real integration skips."""
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
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ('phase212','phase213','phase30','phase31')
        for p in (ROOT/'outputs'/name).rglob('*') if p.is_file()}

def main():
    started=time.perf_counter();before=snapshot();out=ROOT/'outputs/phase31b'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'tests.log').open('w',encoding='utf-8') as stream,redirect_stdout(stream),redirect_stderr(stream):
        suite=unittest.TestSuite()
        for p in sorted((ROOT/'tests').glob('test_*.py')):
            if p.name!='test_phase31b.py':suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p.name))
        old=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        new=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_phase31b.py'))
    if not old.wasSuccessful() or old.testsRun!=216 or not new.wasSuccessful():
        raise SystemExit('Tests failed; see outputs/phase31b/tests.log')
    if not all(compileall.compile_dir(ROOT/f,quiet=1) for f in ('src','scripts','tests')):raise SystemExit('Syntax failure')
    for m in pkgutil.iter_modules([str(ROOT/'src')]):importlib.import_module('src.'+m.name)
    if before!=snapshot():raise SystemExit('Historical artifacts changed')
    protocol=json.loads((ROOT/'outputs/phase213/protocol.json').read_text(encoding='utf-8'))
    for field in ('phase212_snapshot_sha256','new_code_sha256'):
        for path,digest in protocol[field].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise SystemExit('Frozen source binding changed')
    report={'original_tests':old.testsRun,'original_passed':old.testsRun-len(old.skipped),
        'new_tests':new.testsRun,'new_passed':new.testsRun-len(new.skipped),
        'new_skipped':len(new.skipped),'skip_reasons':[reason for test,reason in new.skipped],
        'total_tests':old.testsRun+new.testsRun,'passed':True,'syntax_passed':True,'imports_passed':True,
        'historical_artifacts_unchanged':True,'frozen_bindings_verified':True,
        'real_binding_checks_executed':not any('Real MiniZero binding unavailable' in reason for test,reason in new.skipped),
        'player_id_evaluation':False,'elapsed_seconds':time.perf_counter()-started}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
