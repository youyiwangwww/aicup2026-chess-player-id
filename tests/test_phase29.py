"""Stability-only isolation, frozen protocol and paired uncertainty tests."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.phase29_validation import FIXED, CHECKPOINT, validate_fixed, stability_only, make_stability, paired_bootstrap, readiness
from src.utils import ROOT, load_config


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'configs/phase29.yaml')
        self.selected = load_config(ROOT/'configs/phase28_selected.yaml')

    def test_closed_tests_and_saved_closed_results_never_open(self):
        with stability_only():
            for name in ['outputs/splits/test_ground_truth.csv', 'outputs/splits/test_candidates.csv',
                'outputs/round1_archive/anything.csv', 'outputs/phase26/splits/final_test2_queries.csv',
                'outputs/phase26/final_test2/random_predictions.csv']:
                with self.assertRaises(PermissionError):
                    (ROOT/'.test-tmp/phase29_guard'/name).read_bytes()

    def test_other_dev2_checkpoints_and_raw_truth_blocked(self):
        with stability_only():
            for name in ['outputs/phase28/splits/val_ground_truth.csv',
                         'outputs/phase28/experiments/Triplet-Hard-200/best.pt']:
                with self.assertRaises(PermissionError):
                    (ROOT/'.test-tmp/phase29_guard'/name).read_bytes()

    def test_alpha_fixed_point_nine(self):
        validate_fixed(self.config, self.selected)
        for alpha in [0, .8, 1]:
            config = copy.deepcopy(self.config)
            config['fixed']['alpha'] = alpha
            with self.assertRaises(ValueError):
                validate_fixed(config, self.selected)

    def test_window_fixed_player_five_color_aware(self):
        for key, value in [('opening_mode', 'total'), ('opening_window', 10), ('color_aware', False)]:
            config = copy.deepcopy(self.config)
            config['fixed'][key] = value
            with self.assertRaises(ValueError):
                validate_fixed(config, self.selected)

    def test_checkpoint_and_epoch_cannot_be_selected_again(self):
        for key, value in [('best_epoch', 19), ('checkpoint', 'outputs/phase28/experiments/Triplet-Hard-100/last.pt'),
                           ('experiment', 'Triplet-Hard-200')]:
            selected = copy.deepcopy(self.selected)
            selected['selection']['triplet'][key] = value
            with self.assertRaises(ValueError):
                validate_fixed(self.config, selected)

    def test_excluded_dev2_identities_cannot_enter_split(self):
        source = pd.DataFrame([{'player_id': p, 'game_id': p+str(i), 'color': 'B',
            'sgf_content': f'(;SZ[19]C[{p}{i}];B[aa])', 'rank': '1d'}
            for p in ['old_test', 'dev2_train', 'dev2_val', 'new1', 'new2', 'new3'] for i in range(4)])
        excluded = {'old_test', 'dev2_train', 'dev2_val'}
        settings = {'players': 2, 'candidate_games': 2, 'query_games': 1}
        c, q, t, audit = make_stability(source, excluded, settings, 42)
        self.assertFalse(set(c.player_id)&excluded)
        self.assertFalse(set(c.game_id)&set(q.game_id))
        self.assertFalse(set(c.sgf_content)&set(q.sgf_content))
        self.assertNotIn('player_id', q)
        self.assertEqual(audit['max_eligible_players'], 3)
        for a, b in zip([c, q, t], make_stability(source, excluded, settings, 42)[:3]):
            pd.testing.assert_frame_equal(a, b)

    def test_insufficient_data_reports_maximum_and_stops(self):
        source = pd.DataFrame({'player_id': ['a'], 'game_id': ['g'], 'sgf_content': ['sgf']})
        with self.assertRaisesRegex(ValueError, 'maximum eligible=0'):
            make_stability(source, set(), {'players': 100, 'candidate_games': 30, 'query_games': 10}, 42)

    def test_bootstrap_deterministic_and_question_order_invariant(self):
        a = pd.DataFrame({'question_id': ['q1', 'q2', 'q3'], 'competition_score': [1., 0., .3]})
        b = pd.DataFrame({'question_id': ['q1', 'q2', 'q3'], 'competition_score': [1., .2, .3]})
        rows, samples = paired_bootstrap(a, b)
        again, same = paired_bootstrap(a.iloc[::-1], b.iloc[::-1])
        self.assertEqual(rows, again)
        pd.testing.assert_frame_equal(samples, same)
        self.assertEqual(len(samples), 1000)

    def test_bootstrap_preserves_pairing_and_requires_same_questions(self):
        a = pd.DataFrame({'question_id': ['q1', 'q2'], 'competition_score': [1., 0.]})
        rows, _ = paired_bootstrap(a, a.copy())
        self.assertEqual(rows[2]['ci_lower'], 0)
        self.assertEqual(rows[2]['ci_upper'], 0)
        bad = a.copy()
        bad.loc[0, 'question_id'] = 'unknown'
        with self.assertRaises(ValueError):
            paired_bootstrap(a, bad)

    def test_readiness_does_not_require_difference_ci_excludes_zero(self):
        self.assertEqual(readiness(.7, .71, .01, 100, .6)['status'], 'READY FOR FINAL TEST 3')
        self.assertEqual(readiness(.7, .69, -.01, 100, .6)['status'], 'NOT READY FOR FINAL TEST 3')

    def test_no_training_optimizer_or_backward_in_pipeline(self):
        import torch
        from src.phase28_common import encoder
        from src.phase28_triplet import score_embeddings
        from tests.test_phase28 import FakeStore
        model = encoder({'model': {'in_channels': 17, 'channels': 4, 'num_blocks': 1,
                                  'embedding_dim': 4, 'pool_size': 3}}, 'cpu')
        c, q = FakeStore(2), FakeStore(2)
        for r in q.records:
            r['group_id'] = 'q'+r['group_id'][1:]
        with patch('torch.optim.AdamW', side_effect=AssertionError('no optimizer')), \
             patch('torch.Tensor.backward', side_effect=AssertionError('no backward')):
            scores, _, _ = score_embeddings(model, c, q, 'cpu', 4, ['p000', 'p001'], ['q000', 'q001'])
        self.assertEqual(scores.shape, (2, 2))
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
