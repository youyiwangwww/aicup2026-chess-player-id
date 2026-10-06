"""DEV3 data gate and historical isolation tests, without any training or official fixtures."""
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from scripts.phase212_eligibility import dev3_capacity
from src.phase212_guard import dev3_only


class Phase212EligibilityTests(unittest.TestCase):
    def test_val_players_cannot_be_counted_again_as_train(self):
        counts=pd.Series([40]*54+[20]*121+[19]*30)
        audit=dev3_capacity(counts)
        self.assertEqual(audit['eligible_at_least_20_games'],175)
        self.assertEqual(audit['eligible_at_least_40_games'],54)
        self.assertEqual(audit['maximum_train_players_after_50_val'],125)
        self.assertFalse(audit['can_build_requested_dev3'])
        self.assertEqual(audit['requested_train_players'],150)
        self.assertEqual(audit['requested_val_players'],50)

    def test_exact_disjoint_budget_can_pass(self):
        audit=dev3_capacity(pd.Series([40]*50+[20]*150))
        self.assertEqual(audit['maximum_train_players_after_50_val'],150)
        self.assertTrue(audit['can_build_requested_dev3'])

    def test_many_train_players_do_not_replace_required_val_players(self):
        audit=dev3_capacity(pd.Series([40]*49+[20]*300))
        self.assertFalse(audit['can_build_requested_dev3'])
        self.assertIsNone(audit['maximum_train_players_after_50_val'])

    def test_all_historical_data_scores_and_checkpoints_denied(self):
        paths=['outputs/splits/train.csv','outputs/splits/test_candidates.csv',
               'outputs/splits/test_queries.csv','outputs/splits/test_ground_truth.csv',
               'outputs/phase26/splits/final_test2_candidates.csv','outputs/phase26/final_test2/triplet_scores.npz',
               'outputs/phase210/splits/ground_truth.csv','outputs/phase210/fusion_scores.npz',
               'outputs/phase29/splits/ground_truth.csv','outputs/phase28/splits/val_ground_truth.csv',
               'outputs/phase28/experiments/Triplet-Hard-100/best.pt','outputs/phase211/summary.json']
        with tempfile.TemporaryDirectory() as folder,dev3_only():
            for path in paths:
                with self.assertRaises(PermissionError,msg=path):
                    (Path(folder)/path).open('rb')

    def test_copied_identity_metadata_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'outputs/phase210/inference_complete.json'
            path.parent.mkdir(parents=True)
            path.write_text('{"players":[]}')
            with dev3_only():
                self.assertIn('players',path.read_text())
                with self.assertRaises(PermissionError):
                    path.write_text('modified')


if __name__=='__main__':
    unittest.main()
