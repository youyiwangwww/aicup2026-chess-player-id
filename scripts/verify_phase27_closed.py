"""Verify closure by testing rejected commands, without evaluating TEST again."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metric_utils import file_digest, write_json


def main():
    """Check receipt, fixed hashes, and real CLI refusal with preserved artifacts."""
    directory = ROOT / 'outputs/phase26/final_test2'
    path = directory / 'final_test2_receipt.json'
    before = file_digest(path)
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt['evaluation_count'] != 1 or receipt['status'] != 'CLOSED TEST':
        raise ValueError('Expected exactly one successful closed evaluation')
    if (file_digest(ROOT / 'configs/phase26_selected.yaml') != receipt['config_sha256']
            or file_digest(ROOT / 'outputs/phase26/selected/best.pt') != receipt['checkpoint_sha256']):
        raise ValueError('Frozen config or checkpoint changed')
    commands = [(['-m', 'src.phase27_final_test2'], 2),
                (['-m', 'src.player_features', '--config', 'configs/phase26_selected.yaml'], 1),
                (['-m', 'src.run_phase26'], 1)]
    checks = []
    for command, expected in commands:
        result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True,
                                text=True, encoding='utf-8', errors='replace')
        if result.returncode != expected or 'CLOSED' not in result.stderr:
            raise ValueError(f'Replay was not correctly rejected: {command}, {result.stderr}')
        if 'ONE-SHOT RESULT' in result.stdout:
            raise ValueError('A replay unexpectedly emitted TEST results')
        checks.append({'command': 'python ' + ' '.join(command), 'exit_code': result.returncode,
                       'rejected': True, 'evaluation_performed': False})
    if file_digest(path) != before:
        raise ValueError('Replay modified the final receipt')
    if (file_digest(ROOT / 'configs/phase26_selected.yaml') != receipt['config_sha256']
            or file_digest(ROOT / 'outputs/phase26/selected/best.pt') != receipt['checkpoint_sha256']):
        raise ValueError('Replay modified a frozen artifact')
    write_json({'evaluation_count': 1, 'receipt_unchanged': True, 'config_unchanged': True,
                'checkpoint_unchanged': True, 'checks': checks}, directory / 'closure_verification.json')
    print('Closure verified: final evaluation, legacy preprocessing, and Phase 2.6 rerun all refused. No second evaluation.')


if __name__ == '__main__':
    main()
