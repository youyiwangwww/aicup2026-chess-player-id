"""One-shot lifecycle, leakage, and inference-only tests using isolated fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from src import final_test2_lock as lock
from src import phase27_final_test2 as runner
from src.phase27_preflight import check_final_counts, overlap_audit, preflight
from src.metric_utils import write_json


class TinyInference(torch.nn.Module):
    """Fixture encoder asserts eval mode and disabled gradient recording."""
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 2)

    def forward(self, features):
        assert not self.training and not torch.is_grad_enabled()
        values = torch.stack([features.sum(dim=(1, 2, 3)) + 1, torch.ones(len(features))], dim=1)
        return torch.nn.functional.normalize(values, dim=1)


class FinalTest2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.test-tmp')
        self.root = Path(self.temporary.name)
        self.patch_root = patch.object(lock, 'ROOT', self.root)
        self.patch_root.start()

    def tearDown(self):
        self.patch_root.stop()
        self.temporary.cleanup()

    def fixture(self):
        """Build small test-only games; never create or overwrite official data."""
        c = pd.DataFrame([{'player_id': p, 'game_id': f'{p}-c{i}', 'color': 'B',
             'sgf_content': f'(;SZ[19]C[{p}-c{i}];B[dd];W[pp];B[pd];W[dp])'}
             for p in ['p0', 'p1'] for i in range(2)])
        q = pd.DataFrame([{'question_id': f'q{i}', 'game_id': f'q{i}', 'color': 'W',
             'sgf_content': f'(;SZ[19]C[q{i}];B[dd];W[pp];B[pd];W[dp])'} for i in range(2)])
        truth = pd.DataFrame({'question_id': ['q0', 'q1'], 'player_id': ['p0', 'p1']})
        paths = {}
        for name, frame in [('test_candidates_csv', c), ('test_queries_csv', q)]:
            path = self.root / f'{name}.csv'
            frame.to_csv(path, index=False)
            paths[name] = str(path)
        paths['best_checkpoint'] = str(self.root / 'fixture.pt')
        config = {'seed': 42, 'paths': paths,
                  'features': {'board_size': 19, 'history_length': 8, 'max_positions_per_game': 16},
                  'cache': {'games_per_shard': 2, 'max_cached_shards': 2},
                  'training': {'device': 'cpu', 'deterministic': True, 'num_threads': 1},
                  'inference': {'batch_size': 4}, 'model': {'fixture': True}}
        safety = {'status': 'PASSED', 'config_sha256': 'fixture', 'checkpoint_sha256': 'fixture',
                  'dataset_sha256': 'fixture', 'split_sha256': {}, 'selected_experiment': 'H20-Hard', 'best_epoch': 17}
        return config, c, q, truth, safety

    def test_fixed_counts_and_query_identity_are_validated(self):
        _, c, q, truth, _ = self.fixture()
        labeled = check_final_counts(c, q, truth, players=2, candidate_games=2, query_games=1)
        self.assertEqual(list(labeled.player_id), ['p0', 'p1'])
        with self.assertRaisesRegex(ValueError, 'games/player'):
            check_final_counts(c.iloc[:-1], q, truth, players=2, candidate_games=2, query_games=1)
        with self.assertRaisesRegex(ValueError, 'one query'):
            check_final_counts(c, q, truth.assign(player_id='p0'), players=2, candidate_games=2, query_games=1)

    def test_overlap_audit_rejects_identity_game_and_sgf_leakage(self):
        a = pd.DataFrame({'player_id': ['a'], 'game_id': ['a'], 'sgf_content': ['sgf-a']})
        b = pd.DataFrame({'player_id': ['b'], 'game_id': ['b'], 'sgf_content': ['sgf-b']})
        for column in ['player_id', 'game_id', 'sgf_content']:
            with self.subTest(column=column), self.assertRaises(ValueError):
                overlap_audit({'train': a, 'final_candidate': b.assign(**{column: a[column].iloc[0]})})
        self.assertEqual(overlap_audit({'train': a, 'final_candidate': b})['train/final_candidate']['player_id'], 0)

    def test_unfrozen_selection_stops_before_dataset_access(self):
        path = self.root / 'configs/phase26_selected.yaml'
        path.parent.mkdir()
        path.write_text('selection:\n  frozen: false\n', encoding='utf-8')
        with patch('src.phase27_preflight.ROOT', self.root), \
             patch('src.phase27_preflight.resolve_path', lambda p: self.root / Path(p)), \
             patch('src.phase27_preflight.read_csv', side_effect=AssertionError('must not read data')):
            with self.assertRaisesRegex(ValueError, 'not frozen'):
                preflight()

    def test_atomic_claim_refuses_concurrent_and_failed_retries(self):
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with lock.claim_once():
                with self.assertRaises(ValueError):
                    with lock.claim_once():
                        pass
                raise RuntimeError('fixture failure')
        self.assertEqual(json.loads((lock.final_directory() / 'one_shot_attempt.json').read_text())['status'], 'FAILED_CLOSED')
        with self.assertRaises(ValueError):
            with lock.claim_once():
                pass

    def test_closed_inputs_block_actual_open_and_shared_csv_reader(self):
        path = self.root / 'outputs/phase26/splits/final_test2_ground_truth.csv'
        path.parent.mkdir(parents=True)
        path.write_text('question_id,player_id\nq,p\n', encoding='utf-8')
        with lock.claim_once():
            self.assertIn('q,p', path.read_text())
        from src.utils import read_csv
        for opener in [lambda: path.read_text(), lambda: open(path),
                       lambda: read_csv(path, ['question_id', 'player_id'])]:
            with self.assertRaises(ValueError):
                opener()

    def test_frozen_config_and_checkpoint_writes_are_blocked_after_claim(self):
        paths = [self.root / 'configs/phase26_selected.yaml', self.root / 'outputs/phase26/selected/best.pt']
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'unchanged')
        replacement = self.root / 'replacement.tmp'
        replacement.write_bytes(b'changed')
        with lock.claim_once():
            for path in paths:
                with self.assertRaises(PermissionError):
                    path.write_bytes(b'changed')
                with self.assertRaises(PermissionError):
                    replacement.replace(path)
                self.assertEqual(path.read_bytes(), b'unchanged')

    def test_closed_summary_alone_rejects_replay(self):
        path = self.root / 'outputs/phase26/summary.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"final_test2_evaluations": 1}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'summary'):
            lock.require_unconsumed()

    def test_orchestrator_inference_only_closes_once_and_keeps_config(self):
        fixture = self.fixture()
        config = fixture[0]
        checkpoint = {'epoch': 17, 'model_config': config['model']}
        def isolated_json(value, path):
            return write_json(value, self.root / Path(path))
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'preflight', return_value=fixture), \
             patch.object(runner, 'verify_unchanged'), patch.object(runner, 'file_digest', return_value='fixture'), \
             patch.object(runner, 'load_encoder', return_value=(TinyInference(), checkpoint)), \
             patch.object(runner, 'read_json', return_value={'status': 'completed', 'final_test2_evaluations': 0}), \
             patch.object(runner, 'write_json', side_effect=isolated_json), \
             patch('torch.Tensor.backward', side_effect=AssertionError('backprop forbidden')), \
             patch('torch.optim.AdamW', side_effect=AssertionError('optimizer forbidden')):
            receipt = runner.run_once()
            self.assertEqual(receipt['evaluation_count'], 1)
            self.assertEqual(receipt['status'], 'CLOSED TEST')
            self.assertEqual(set(receipt['metrics']), {'random', 'opening10', 'triplet_hard'})
            self.assertEqual(receipt['opening_total_moves'], 10)
            self.assertFalse(receipt['training_performed'])
            comparison = pd.read_csv(lock.final_directory() / 'final_comparison.csv')
            self.assertEqual(list(comparison.columns), ['method', 'top1', 'top3', 'top5', 'competition_score'])
            before = (lock.final_directory() / 'final_test2_receipt.json').read_bytes()
            with patch.object(runner, 'preprocess_partition', side_effect=AssertionError('repeat preprocessing forbidden')):
                with self.assertRaises(ValueError):
                    runner.run_once()
            self.assertEqual(before, (lock.final_directory() / 'final_test2_receipt.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
