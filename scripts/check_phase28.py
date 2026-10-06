"""Verify all project tests/imports/syntax without opening CLOSED TEST truths."""
import compileall
from contextlib import redirect_stdout, redirect_stderr
import importlib
import json
from pathlib import Path
import pkgutil
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metric_utils import file_digest, write_json


def main():
    """Keep complete checks and protect the historical frozen configuration/weights."""
    started = time.perf_counter()
    directory = ROOT/'outputs/phase28'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'tests.log').open('w', encoding='utf-8') as stream, redirect_stdout(stream), redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    syntax = all(compileall.compile_dir(ROOT/name, quiet=1) for name in ['src', 'scripts', 'tests'])
    imported = []
    for module in pkgutil.iter_modules([str(ROOT/'src')]):
        importlib.import_module('src.'+module.name)
        imported.append(module.name)
    expected = {'data/training/train_A.csv': '0915826dfeadb32fa6fc5f5a08b1ef90a5ed0e3cc7868452e8158c887e3ccbca',
        'configs/phase26_selected.yaml': '7508ff83002fd6ae87f4a6384e58245a49eef12df2ef3a25b80e44a40f9c4ec1',
        'outputs/phase26/selected/best.pt': '2807c60a056e2325c70e2b35065ca35fdd836eecdcbad9e25fdf9418bbabd4e5'}
    hashes = {path: file_digest(ROOT/path) for path in expected}
    immutable = hashes == expected
    # Ledger metadata only; never read any CLOSED TEST ground-truth/input file.
    receipt = json.loads((ROOT/'outputs/phase26/final_test2/final_test2_receipt.json').read_text(encoding='utf-8'))
    closed = receipt['evaluation_count'] == 1 and receipt['status'] == 'CLOSED TEST'
    report = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'skipped': len(result.skipped), 'syntax_passed': syntax, 'imports_passed': True, 'imported_modules': imported,
        'historical_frozen_artifacts_unchanged': immutable, 'sha256': hashes, 'final_test2_still_closed': closed,
        'elapsed_seconds': time.perf_counter()-started}
    write_json(report, directory/'verification.json')
    print(json.dumps({k: v for k, v in report.items() if k not in ['imported_modules', 'sha256']}, indent=2))
    return 0 if result.wasSuccessful() and syntax and immutable and closed else 1


if __name__ == '__main__':
    sys.exit(main())
