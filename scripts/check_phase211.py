"""Run the full synthetic unit suite plus import/syntax checks without real TEST reads."""
import compileall
from contextlib import redirect_stdout,redirect_stderr
import importlib
from pathlib import Path
import pkgutil
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.metric_utils import write_json


def main():
    """Isolate verification outputs; existing TEST unit fixtures are synthetic only."""
    start=time.perf_counter()
    directory=ROOT/'outputs/phase211'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'tests.log').open('w',encoding='utf-8') as stream,redirect_stdout(stream),redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    syntax=all(compileall.compile_dir(ROOT/name,quiet=1) for name in ['src','scripts','tests'])
    modules=[]
    for module in pkgutil.iter_modules([str(ROOT/'src')]):
        importlib.import_module('src.'+module.name)
        modules.append(module.name)
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'syntax_passed':syntax,'imports_passed':True,'imported_modules':modules,'elapsed_seconds':time.perf_counter()-start}
    write_json(report,directory/'verification.json')
    print({k:v for k,v in report.items() if k!='imported_modules'})
    return 0 if result.wasSuccessful() and syntax else 1


if __name__=='__main__':
    sys.exit(main())
