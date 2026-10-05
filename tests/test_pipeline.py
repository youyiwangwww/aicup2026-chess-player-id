"""Meaningful correctness checks without any real training data."""
import importlib
import math
import unittest

import numpy as np
import pandas as pd

from src.analyze_dataset import analyze_dataset
from src.baseline import predict
from src.build_validation import build_validation
from src.evaluate import evaluate
from src.opening_features import build_fingerprints, game_heatmap
from src.sgf_parser import parse_target_moves


def sample_frame():
    """Provide enough unique, valid games for two players."""
    return pd.DataFrame([
        {'player_id': str(p), 'game_id': f'{p}-{g}', 'rank': 'A', 'color': 'B',
         'sgf_content': f'(;SZ[19]C[{p}-{g}];B[{"aa" if p == 0 else "ss"}];W[cc])'}
        for p in range(2) for g in range(6)])


class PipelineTests(unittest.TestCase):
    def test_imports(self):
        for name in ['utils', 'analyze_dataset', 'build_validation', 'sgf_parser',
                     'opening_features', 'baseline', 'evaluate', 'create_mock_data']:
            importlib.import_module(f'src.{name}')

    def test_color_pass_setup_and_main_variation(self):
        content = '(;SZ[19]AB[cc];B[aa];W[bb];B[];W[dd](;B[ee])(;B[ff]))'
        self.assertEqual(parse_target_moves(content, 'B', 3), [(18, 0)])
        self.assertEqual(parse_target_moves(content, 'W', 3), [(17, 1)])
        self.assertEqual(parse_target_moves(content, 'B', 5), [(18, 0), (14, 4)])
        self.assertEqual(game_heatmap(content, 'W', 4).sum(), 2)

    def test_invalid_and_empty_games_are_skipped(self):
        frame = sample_frame().iloc[:1].copy()
        broken = frame.assign(game_id='bad', sgf_content='invalid')
        empty = frame.assign(game_id='empty', sgf_content='(;SZ[19];B[])')
        blank = frame.assign(game_id='blank', sgf_content='')
        invalid_color = frame.assign(game_id='color', color='')
        fp, errors = build_fingerprints(pd.concat([frame, broken, empty, blank, invalid_color]), 'player_id')
        self.assertEqual(len(errors), 4)
        self.assertEqual(len(fp), 1)
        self.assertEqual(fp['0'].sum(), 1)
        with self.assertRaises(ValueError):
            parse_target_moves('(;SZ[9];B[aa])', 'B')
        with self.assertRaises(ValueError):
            parse_target_moves('(;SZ[19];B[aa])', 'unknown')

    def test_split_reproducibility_disjointness_and_no_label(self):
        frame = sample_frame()
        first = build_validation(frame, 2, 3, 2, 42)
        second = build_validation(frame.sample(frac=1, random_state=9), 2, 3, 2, 42)
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)
        candidates, queries, truth = first
        self.assertFalse(set(candidates.game_id) & set(queries.game_id))
        self.assertFalse(set(candidates.sgf_content) & set(queries.sgf_content))
        self.assertNotIn('player_id', queries.columns)
        self.assertEqual(queries.groupby('question_id').size().tolist(), [2, 2])
        self.assertEqual(len(truth), 2)
        with self.assertRaises(ValueError):
            build_validation(pd.concat([frame, frame]), 2, 6, 1, 42)

    def test_shared_games_do_not_cross_split(self):
        frame = sample_frame()
        shared = frame[frame.player_id == '0'].assign(player_id='1')
        candidates, queries, _ = build_validation(pd.concat([frame, shared]), 2, 2, 1, 42)
        self.assertFalse(set(candidates.game_id) & set(queries.game_id))
        self.assertFalse(set(candidates.sgf_content) & set(queries.sgf_content))

    def test_end_to_end(self):
        candidates, queries, truth = build_validation(sample_frame(), 2, 3, 2, 42)
        cfp, _ = build_fingerprints(candidates, 'player_id')
        qfp, _ = build_fingerprints(queries, 'question_id')
        predictions = predict(cfp, qfp)
        metrics, _ = evaluate(predictions, truth)
        self.assertTrue(all(value == 1 for value in metrics.values()))
        self.assertEqual(analyze_dataset(sample_frame()).iloc[0]['value'], 12)

    def test_scores_all_ranks_and_missing_question(self):
        truth = pd.DataFrame([{'question_id': f'q{i}', 'player_id': 'correct'} for i in range(1, 7)])
        rows = []
        for i in range(1, 6):
            for rank in range(1, 6):
                rows.append({'question_id': f'q{i}', 'rank': rank,
                             'player_id': 'correct' if rank == i else f'wrong{rank}'})
        metrics, details = evaluate(pd.DataFrame(rows), truth)
        self.assertAlmostEqual(metrics['top_1_accuracy'], 1 / 6)
        self.assertAlmostEqual(metrics['top_3_accuracy'], 3 / 6)
        self.assertAlmostEqual(metrics['top_5_accuracy'], 5 / 6)
        self.assertAlmostEqual(metrics['competition_score'], sum(math.exp(-i) for i in range(5)) / 6)
        self.assertEqual(details.iloc[-1]['competition_score'], 0)
        empty = pd.DataFrame(columns=['question_id', 'rank', 'player_id'])
        self.assertEqual(evaluate(empty, truth)[0]['competition_score'], 0)

    def test_ties_are_deterministic(self):
        result = predict({'b': np.ones(361), 'a': np.ones(361)}, {'q': np.ones(361)})
        self.assertEqual(result.player_id.tolist(), ['a', 'b'])

    def test_invalid_rankings_are_rejected(self):
        truth = pd.DataFrame([{'question_id': 'q', 'player_id': 'a'}])
        for rows in [
            [{'question_id': 'unknown', 'player_id': 'a', 'rank': 1}],
            [{'question_id': 'q', 'player_id': 'a', 'rank': 2}],
            [{'question_id': 'q', 'player_id': 'a', 'rank': 1},
             {'question_id': 'q', 'player_id': 'a', 'rank': 2}],
            [{'question_id': 'q', 'player_id': 'a', 'rank': 1.5}],
        ]:
            with self.assertRaises(ValueError):
                evaluate(pd.DataFrame(rows), truth)


if __name__ == '__main__':
    unittest.main()
