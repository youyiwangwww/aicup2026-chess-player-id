"""Focused DEV2 safeguards, windows, exposure and metric checks."""
from collections import Counter
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.phase28_common import DeterministicAdaptiveAverage, dev2_only, encoder, fuse, normalize_scores, rankings
from src.phase28_opening import fingerprints, opening_scores, player_moves
from src.phase28_split import color_cohort, make_dev2, exclusion_ids
from src.phase28_triplet import ExposureTriplets, diagnostics, hard_negative, train_experiment
from src.player_model import PlayerEncoder
from src.utils import ROOT, write_csv


class FakeStore:
    """Minimal in-memory TRAIN fixtures; no official or closed data."""
    def __init__(self, players=3):
        self.records = [{'group_id': f'p{i:03d}', 'game_id': f'{i}_{j}', 'color': 'B' if j < 2 else 'W',
                         'num_positions': 2} for i in range(players) for j in range(4)]

    def features(self, index):
        return np.zeros((2, 17, 19, 19), dtype=np.uint8)

    def __len__(self):
        return len(self.records)


def source_fixture():
    """Provide distinct synthetic IDs/SGFs solely for unit-level split tests."""
    return pd.DataFrame([{'player_id': f'p{i}', 'game_id': f'g{i}_{j}', 'rank': '1d',
                         'color': 'B' if j % 2 == 0 else 'W',
                         'sgf_content': f'(;SZ[19]C[{i}_{j}];B[aa];W[bb])'}
                        for i in range(6) for j in range(8)])


class Phase28Tests(unittest.TestCase):
    def test_target_move_window(self):
        sgf = '(;SZ[19];B[aa];W[bb];B[];W[cc];B[dd])'
        self.assertEqual(len(player_moves(sgf, 'B', 2)), 1)
        self.assertEqual(len(player_moves(sgf, 'B', 3)), 2)
        self.assertEqual(len(player_moves(sgf, 'W', 1)), 1)

    def test_invalid_sgf_is_per_game_failure(self):
        frame = pd.DataFrame([{'player_id': 'p', 'color': 'B', 'sgf_content': 'bad', 'game_id': 'g'}])
        _, banks, errors = fingerprints(frame, 'player_id', 'player', 5)
        self.assertEqual(len(errors), 1)

    def color_frames(self):
        candidates = pd.DataFrame([{'player_id': p, 'color': c, 'game_id': p+c,
            'sgf_content': f'(;SZ[19];B[{b}];W[{w}])'}
            for p, b, w in [('p0', 'dd', 'pp'), ('p1', 'pp', 'dd')] for c in ['B', 'W']])
        queries = pd.DataFrame([{'question_id': 'q', 'color': c, 'game_id': f'q{i}',
            'sgf_content': '(;SZ[19];B[pp];W[pp])'} for i, c in enumerate(['W', 'W', 'B'])])
        return candidates, queries

    def test_black_white_banks_separate(self):
        c, _ = self.color_frames()
        _, banks, _ = fingerprints(c, 'player_id', 'player', 1)
        self.assertFalse(np.array_equal(banks['p0', 'B'], banks['p0', 'W']))

    def test_query_color_matching_and_weight(self):
        c, q = self.color_frames()
        scores, *_ = opening_scores(c, q, 'player', 1, True)
        np.testing.assert_allclose(scores, [[2/3, 1/3]])
        scores, *_ = opening_scores(c, q.iloc[:1], 'player', 1, True)
        np.testing.assert_allclose(scores, [[1, 0]])

    def test_fusion_unique_top5(self):
        scores = fuse(np.arange(7)[None, :], -np.arange(7)[None, :], .4)
        result = rankings(scores, [str(i) for i in range(7)], ['q'])
        self.assertEqual(result.player_id.nunique(), 5)

    def test_alpha_zero_equals_triplet(self):
        a, b = np.array([[1., 3., 2.]]), np.array([[.99, .99001, .99002]])
        for norm in ['z-score', 'min-max']:
            self.assertEqual(rankings(fuse(a, b, 0, norm), ['a', 'b', 'c'], ['q']).player_id.tolist(),
                             rankings(b, ['a', 'b', 'c'], ['q']).player_id.tolist())

    def test_alpha_one_equals_opening(self):
        a, b = np.array([[1., 3., 2.]]), np.array([[.99, .99001, .99002]])
        for norm in ['z-score', 'min-max']:
            self.assertEqual(rankings(fuse(a, b, 1, norm), ['a', 'b', 'c'], ['q']).player_id.tolist(),
                             rankings(a, ['a', 'b', 'c'], ['q']).player_id.tolist())

    def test_normalization_per_question_flat_and_ties(self):
        a = np.array([[1., 1., 1.], [2., 3., 5.]])
        n = normalize_scores(a)
        np.testing.assert_array_equal(n[0], 0)
        self.assertAlmostEqual(n[1].mean(), 0)
        self.assertAlmostEqual(n[1].std(), 1)

    def test_closed_test_actual_open_refused(self):
        with dev2_only():
            for name in ['outputs/splits/test_ground_truth.csv',
                         'outputs/phase26/splits/final_test2_ground_truth.csv',
                         'outputs/phase26/final_test2/triplet_predictions.csv']:
                with self.assertRaises(PermissionError):
                    (Path('.test-tmp/phase28_guard') / name).read_bytes()

    def test_exclusion_recovery_uses_metadata_columns_only(self):
        def metadata(path):
            if path.endswith('final_test2_receipt.json'):
                return {'evaluation_count': 1, 'status': 'CLOSED TEST', 'final_players': 50}
            self.assertEqual(path, 'outputs/phase26/splits/split_audit.json')
            return {'closed_round1_excluded_players': [f'old{i}' for i in range(10)]}
        def identities(path, **kwargs):
            self.assertEqual(kwargs['usecols'], ['player_id'])
            if Path(path).name == 'dev_val_candidates.csv':
                ids = [f'val{i}' for i in range(50)]
            else:
                self.assertIn(Path(path).name, ['random_predictions.csv', 'opening10_predictions.csv', 'triplet_predictions.csv'])
                ids = [f'final{i}' for i in range(50)]
            return pd.DataFrame({'player_id': ids})
        with patch('src.phase28_split.read_json', side_effect=metadata), \
             patch('src.phase28_split.pd.read_csv', side_effect=identities), \
             patch('src.phase28_split.file_digest', return_value='sha256'):
            excluded, provenance = exclusion_ids()
        self.assertEqual(len(excluded), 110)
        self.assertEqual(len(provenance['final_test2']), 50)

    def test_dev2_split_disjoint_and_reproducible(self):
        settings = {'train_players': 2, 'val_players': 2, 'train_games': 2, 'candidate_games': 2, 'query_games': 1}
        frames, _, audit = make_dev2(source_fixture(), {'p0'}, settings)
        again, _, _ = make_dev2(source_fixture(), {'p0'}, settings)
        for a, b in zip(frames, again):
            pd.testing.assert_frame_equal(a, b)
        self.assertFalse(set(frames[0].player_id) & set(frames[1].player_id))
        self.assertNotIn('player_id', frames[2])
        self.assertEqual(audit['excluded_overlap'], 0)
        self.assertEqual(sum(audit['game_id_overlap'].values()), 0)
        self.assertEqual(sum(audit['sgf_content_overlap'].values()), 0)

    def test_insufficient_split_is_explicit_error(self):
        with self.assertRaises(ValueError):
            make_dev2(source_fixture(), set(), {'train_players': 200, 'val_players': 100,
                'train_games': 20, 'candidate_games': 30, 'query_games': 10})

    def test_matched_colors_same_ids_and_budgets(self):
        conditions, ids = color_cohort(source_fixture(), 2, 1)
        self.assertEqual(len(conditions), 5)
        for _, c, q, t in conditions:
            self.assertEqual(sorted(t.player_id), ids)
            self.assertEqual(set(c.groupby('player_id').size()), {2})
            self.assertEqual(set(q.groupby('question_id').size()), {1})
            self.assertFalse(set(c.game_id) & set(q.game_id))

    def test_exposure_exact_600_2000_4000(self):
        for n, expected in [(30, 600), (100, 2000), (200, 4000)]:
            store = FakeStore(n)
            dataset = ExposureTriplets(store)
            self.assertEqual(len(dataset), expected)
            for epoch in [0, 1]:
                dataset.set_epoch(epoch)
                counts = Counter(store.records[dataset.sample_indices(i)[0][0]]['group_id'] for i in range(len(dataset)))
                self.assertEqual(len(counts), n)
                self.assertEqual(set(counts.values()), {20})

    def test_cross_color_positive_same_id_opposite_color(self):
        store = FakeStore()
        dataset = ExposureTriplets(store, cross_color=True)
        directions = Counter()
        for i in range(len(dataset)):
            (a, _), (p, _), (n, _) = dataset.sample_indices(i)
            aa, pp, nn = [store.records[x] for x in [a, p, n]]
            self.assertEqual(aa['group_id'], pp['group_id'])
            self.assertNotEqual(aa['color'], pp['color'])
            self.assertNotEqual(aa['group_id'], nn['group_id'])
            directions[aa['group_id'], aa['color']] += 1
        self.assertEqual(set(directions.values()), {10})

    def test_cross_color_eligibility_never_falls_back(self):
        store = FakeStore()
        for r in store.records[:4]:
            r['color'] = 'B'
        dataset = ExposureTriplets(store, cross_color=True)
        self.assertNotIn('p000', dataset.players)
        self.assertEqual(len(dataset), 40)

    def test_hard_negative_mask_and_gradient(self):
        a = torch.tensor([[0., 0.], [1., 1.]])
        pool = torch.tensor([[.1, .1], [.9, .9], [2., 2.]], requires_grad=True)
        selected, indices = hard_negative(a, pool, torch.tensor([0, 1]), torch.tensor([0, 1, 2]))
        self.assertEqual(indices.tolist(), [1, 0])
        selected.sum().backward()
        torch.testing.assert_close(pool.grad, torch.tensor([[1., 1.], [1., 1.], [0., 0.]]))

    def test_diagnostics_collapse_and_separation(self):
        cr = [{'group_id': 'a'}, {'group_id': 'b'}]
        qr = [{'group_id': 'q1'}, {'group_id': 'q2'}]
        truth = pd.DataFrame({'question_id': ['q1', 'q2'], 'player_id': ['a', 'b']})
        settings = {'variance_threshold': 1e-8, 'collapse_fraction': .8}
        collapsed, _ = diagnostics(np.ones((2, 4)), np.ones((2, 4)), cr, qr, truth, settings)
        self.assertTrue(collapsed['collapse_warning'])
        good, _ = diagnostics(np.eye(2), np.eye(2), cr, qr, truth, settings)
        self.assertFalse(good['collapse_warning'])
        self.assertEqual(good['separation'], 1)

    def test_equivalent_pool_forward_and_gradient(self):
        for h, w in [(19, 19), (7, 11)]:
            a = torch.randn(2, 3, h, w, dtype=torch.float64, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            ref, actual = torch.nn.AdaptiveAvgPool2d(3)(a), DeterministicAdaptiveAverage(3)(b)
            torch.testing.assert_close(ref, actual, rtol=1e-12, atol=1e-12)
            gradient = torch.randn_like(ref)
            ref.backward(gradient)
            actual.backward(gradient)
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)

    def test_encoder_architecture_and_parameters_unchanged(self):
        settings = {'in_channels': 17, 'channels': 8, 'num_blocks': 1, 'embedding_dim': 4, 'pool_size': 3}
        original, actual = PlayerEncoder(**settings), encoder({'model': settings}, 'cpu')
        self.assertEqual(list(original.state_dict()), list(actual.state_dict()))
        self.assertEqual(sum(p.numel() for p in original.parameters()), sum(p.numel() for p in actual.parameters()))

    def test_fresh_training_checkpoint_and_resume_on_mock_dev_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.test-tmp', prefix='phase28_train_') as temporary:
            directory = Path(temporary)
            relative = directory.relative_to(ROOT).as_posix()
            config = {'seed': 42, 'output_dir': relative, 'paths': {}, 'split': {'train_games': 4},
                'model': {'in_channels': 17, 'channels': 4, 'num_blocks': 1, 'embedding_dim': 4, 'pool_size': 3},
                'features': {'board_size': 19, 'history_length': 8, 'max_positions_per_game': 2},
                'training': {'device': 'cpu', 'epochs': 2, 'batch_size': 4, 'learning_rate': .001,
                    'weight_decay': .0001, 'margin': .2, 'triplets_per_player_per_epoch': 2,
                    'cross_color_min_games': 2, 'num_threads': 1, 'deterministic': True},
                'inference': {'batch_size': 4}, 'diagnostics': {'variance_threshold': 1e-8, 'collapse_fraction': .8}}
            for key in ['val_candidates_csv', 'val_queries_csv', 'val_ground_truth_csv']:
                config['paths'][key] = relative+'/'+key+'.csv'
                write_csv(pd.DataFrame({'id': ['fixture']}), config['paths'][key])
            write_csv(pd.DataFrame({'id': ['fixture']}), directory/'splits/train.csv')
            train, candidates, queries = FakeStore(2), FakeStore(2), FakeStore(2)
            for record in queries.records:
                record['group_id'] = 'q'+record['group_id'][1:]
            truth = pd.DataFrame({'question_id': ['q000', 'q001'], 'player_id': ['p000', 'p001']})
            with dev2_only():
                row, scores, diag, path = train_experiment(config, 'unit-hard', train, candidates, queries,
                    truth, ['p000', 'p001'], ['q000', 'q001'])
                self.assertTrue(path.exists())
                self.assertEqual(row['samples_per_epoch'], 4)
                self.assertIn(row['best_epoch'], [1, 2])
                self.assertEqual(scores.shape, (2, 2))
                with patch('torch.optim.AdamW', side_effect=AssertionError('completed run must not retrain')):
                    again, same, _, _ = train_experiment(config, 'unit-hard', train, candidates, queries,
                        truth, ['p000', 'p001'], ['q000', 'q001'])
                self.assertEqual(again['best_epoch'], row['best_epoch'])
                np.testing.assert_array_equal(scores, same)


if __name__ == '__main__':
    unittest.main()
