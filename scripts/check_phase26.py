"""Run all tests with native Python logging and an unambiguous process exit code."""
import compileall
import json
from pathlib import Path
import sys
import time
import unittest


def main():
    """Save the complete test log and verify all project Python syntax."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    directory = root / '.test-tmp'
    directory.mkdir(exist_ok=True)
    started = time.perf_counter()
    with (directory / 'phase26_tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(root / 'tests')))
    syntax = all(compileall.compile_dir(root / name, quiet=1) for name in ['src', 'scripts', 'tests'])
    report = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'syntax_passed': syntax, 'elapsed_seconds': time.perf_counter() - started}
    (directory / 'phase26_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() and syntax else 1


if __name__ == '__main__':
    sys.exit(main())
