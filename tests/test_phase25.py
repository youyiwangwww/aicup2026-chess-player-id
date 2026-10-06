"""Three-way split, training isolation, final-test lifecycle and reproducibility tests."""
import builtins
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_multi_seed import seed_config, summarize_seeds
from src.analyze_feature_coverage import analyze_coverage, coverage_row
from src.build_metric_split import (THREE_WAY_KEYS, audit_three_way_split,
                                    build_three_way_split)
from src.compare_baselines import compare
from src.create_metric_mock_data import metric_mock_frame
from src.evaluate_test import evaluate_final_test
from src.experiment_state import read_json
from src.experiment_summary import sanitized_config, write_experiment_summary
from src.metric_utils import file_digest, write_json
from src.opening_test import run_opening_test
from src.player_features import preprocess_all
from src.random_baseline import random_predictions, run_random_baseline
from src.run_experiment import MISSING_DATA_MESSAGE, run_experiment
from src.train_triplet import train
from src.triplet_dataset import TripletDataset
from src.utils import ROOT, load_config, resolve_path, write_csv


class ThreeWaySplitTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_config(ROOT / 'configs/phase25_mock.yaml')['split']
        self.frame = metric_mock_frame(14, 6)
        self.parts = build_three_way_split(self.frame, **self.settings, seed=42)

    def test_all_player_pools_disjoint_and_counts(self):
        train_ids = set(self.parts[0].player_id)
        val_ids = set(self.parts[3].player_id)
        test_ids = set(self.parts[6].player_id)
        self.assertFalse(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        audit = self.parts[-1]
        self.assertEqual([audit[f'{name}_player_count'] for name in ['train', 'val', 'test']], [4, 4, 6])
        self.assertEqual([len(frame) for frame in self.parts[:7]], [16, 12, 8, 4, 18, 12, 6])
        self.assertEqual(len(audit['game_id_overlap_counts']), 10)
        self.assertEqual(len(audit['sgf_content_overlap_counts']), 10)
        for category in ['player_overlap_counts', 'game_id_overlap_counts', 'sgf_content_overlap_counts']:
            self.assertTrue(all(value == 0 for value in audit[category].values()))
        self.assertNotIn('player_id', self.parts[2].columns)
        self.assertNotIn('player_id', self.parts[5].columns)

    def test_every_pair_game_and_sgf_leak_is_rejected(self):
        indices = [0, 1, 2, 4, 5]
        for left_index, first in enumerate(indices):
            for second in indices[left_index + 1:]:
                for column in ['game_id', 'sgf_content']:
                    parts = [part.copy() for part in self.parts[:7]]
                    parts[second].loc[0, column] = parts[first].loc[0, column]
                    with self.assertRaisesRegex(ValueError, 'leakage'):
                        audit_three_way_split(*parts, 42)

    def test_identity_leak_is_rejected(self):
        parts = [part.copy() for part in self.parts[:7]]
        player = parts[3].iloc[0].player_id
        old = parts[6].iloc[0].player_id
        parts[4].loc[parts[4].player_id == old, 'player_id'] = player
        parts[6].loc[parts[6].player_id == old, 'player_id'] = player
        with self.assertRaisesRegex(ValueError, 'leakage'):
            audit_three_way_split(*parts, 42)

    def test_seed_is_reproducible_and_counts_not_reduced(self):
        repeat = build_three_way_split(self.frame.sample(frac=1, random_state=11), **self.settings, seed=42)
        for first, second in zip(self.parts[:7], repeat[:7]):
            pd.testing.assert_frame_equal(first, second)
        settings = load_config(ROOT / 'configs/real_quick.yaml')['split']
        with self.assertRaisesRegex(ValueError, 'Need 10 val players'):
            build_three_way_split(self.frame, **settings, seed=42)


class Phase25LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create test-only audited CSVs and caches outside all reported experiments."""
        (ROOT / '.test-tmp').mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT / '.test-tmp')
        cls.directory = Path(cls.temporary.name)
        cls.config = seed_config(load_config(ROOT / 'configs/phase25_mock.yaml'), 42, cls.directory)
        cls.config['paths']['training_csv'] = str(cls.directory / 'source.csv')
        cls.config['training'].update(samples_per_epoch=8, num_threads=1, epochs=2)
        cls.config['model'].update(channels=8, num_blocks=1, embedding_dim=8)
        source = metric_mock_frame(14, 6)
        write_csv(source, cls.config['paths']['training_csv'])
        cls.parts = build_three_way_split(source, **cls.config['split'], seed=42)
        audit = cls.parts[-1]
        for key, frame in zip(THREE_WAY_KEYS, cls.parts[:7]):
            write_csv(frame, cls.config['paths'][key])
        audit['source_csv_sha256'] = file_digest(cls.config['paths']['training_csv'])
        audit['partition_sha256'] = {key: file_digest(cls.config['paths'][key]) for key in THREE_WAY_KEYS}
        write_json(audit, cls.config['paths']['split_audit_json'])
        preprocess_all(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_training_never_reads_test_and_dataset_never_gets_validation_labels(self):
        protected = {resolve_path(value).resolve() for key, value in self.config['paths'].items()
                     if key.startswith('test_')}
        opened, dataset_ids = [], []
        builtin_open = builtins.open
        path_open = Path.open

        def guard(path):
            if isinstance(path, (str, Path)):
                path = Path(path).resolve()
                if (path in protected or (path.parent == resolve_path(self.config['paths']['cache_dir'])
                                          and path.name.startswith('test_'))):
                    raise AssertionError(f'Training accessed TEST: {path.name}')
                opened.append(path.name)

        def guarded_builtin(path, *args, **kwargs):
            guard(path)
            return builtin_open(path, *args, **kwargs)

        def guarded_path(path, *args, **kwargs):
            guard(path)
            return path_open(path, *args, **kwargs)

        def dataset_factory(store, *args, **kwargs):
            ids = {record['group_id'] for record in store.records}
            self.assertEqual(ids, set(self.parts[0].player_id))
            self.assertFalse(ids & set(self.parts[3].player_id))
            self.assertFalse(ids & set(self.parts[6].player_id))
            dataset_ids.append(ids)
            return TripletDataset(store, *args, **kwargs)

        with patch('builtins.open', guarded_builtin), patch.object(Path, 'open', guarded_path), \
                patch('src.train_triplet.TripletDataset', side_effect=dataset_factory):
            train(self.config)
        self.assertEqual(len(dataset_ids), 1)
        self.assertIn('val_ground_truth.csv', opened)
        self.assertNotIn('test_ground_truth.csv', opened)

    def test_best_checkpoint_only_uses_validation_score(self):
        def metric(score):
            return {'top_1_accuracy': 0.0, 'top_3_accuracy': 0.0,
                    'top_5_accuracy': 0.0, 'competition_score': score}, pd.DataFrame()

        with patch('src.train_triplet.evaluate', side_effect=[metric(0.9), metric(0.1)]):
            history = train(self.config)
        best = torch.load(resolve_path(self.config['paths']['best_checkpoint']), weights_only=True)
        self.assertEqual(best['epoch'], 1)
        self.assertEqual(best['best_score'], 0.9)
        self.assertEqual([row['val_competition_score'] for row in history], [0.9, 0.1])
        self.assertTrue({'val_top1', 'val_top3', 'val_top5', 'elapsed_time'} <= history[0].keys())

    def test_final_test_requires_completed_training_and_runs_once(self):
        not_trained = copy.deepcopy(self.config)
        not_trained['paths']['training_state_json'] = str(self.directory / 'not_trained.json')
        with patch('src.evaluate_test.load_store') as store:
            with self.assertRaisesRegex(ValueError, 'Training must complete'):
                evaluate_final_test(not_trained)
            store.assert_not_called()
        train(self.config)
        before = file_digest(self.config['paths']['best_checkpoint'])
        metrics = evaluate_final_test(self.config)
        self.assertTrue(0 <= metrics['competition_score'] <= 1)
        self.assertEqual(file_digest(self.config['paths']['best_checkpoint']), before)
        with patch('src.evaluate_test.retrieve') as retrieval:
            with self.assertRaisesRegex(ValueError, 'already evaluated'):
                evaluate_final_test(self.config)
            retrieval.assert_not_called()

    def test_running_state_or_changed_checkpoint_blocks_test(self):
        altered = copy.deepcopy(self.config)
        altered['paths']['training_state_json'] = str(self.directory / 'running.json')
        write_json({'status': 'running', 'run_id': 'unit-test'}, altered['paths']['training_state_json'])
        with self.assertRaisesRegex(ValueError, 'Training must complete'):
            evaluate_final_test(altered)
        train(self.config)
        checkpoint = resolve_path(self.config['paths']['best_checkpoint'])
        contents = checkpoint.read_bytes()
        try:
            checkpoint.write_bytes(contents + b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                evaluate_final_test(self.config)
        finally:
            checkpoint.write_bytes(contents)

    def test_all_baselines_share_test_questions_and_comparison_does_not_retest(self):
        train(self.config)
        evaluate_final_test(self.config)
        run_random_baseline(self.config)
        run_opening_test(self.config)
        with patch('src.compare_baselines.evaluate', side_effect=AssertionError('Do not re-evaluate test')):
            comparison = compare(self.config)
        self.assertEqual(comparison.method.tolist(), ['random', 'opening', 'triplet'])
        expected = set(self.parts[6].question_id)
        hashes = []
        for key in ['test_predictions_csv', 'random_test_predictions_csv', 'opening_test_predictions_csv']:
            path = resolve_path(self.config['paths'][key])
            predictions = pd.read_csv(path, dtype=str)
            self.assertEqual(set(predictions.question_id), expected)
            self.assertFalse(predictions.duplicated(['question_id', 'player_id']).any())
            self.assertTrue(predictions.groupby('question_id').size().eq(5).all())
            hashes.append(read_json(path.with_suffix('.metadata.json'))['test_source_sha256'])
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[1], hashes[2])
        summary = write_experiment_summary(self.config, 1.25)
        self.assertEqual(summary['dataset_kind'], 'mock')
        self.assertEqual(summary['final_test_evaluations'], 1)
        self.assertEqual(summary['dataset_sha256'], file_digest(self.config['paths']['training_csv']))
        self.assertNotIn(str(ROOT), json.dumps(summary))

    def test_coverage_separates_all_five_partitions(self):
        coverage = analyze_coverage(self.config)
        self.assertEqual(coverage.partition.tolist(), ['train', 'validation_candidate', 'validation_query',
                                                      'test_candidate', 'test_query'])
        self.assertEqual(coverage.total_games.tolist(), [16, 12, 8, 18, 12])
        self.assertTrue(coverage.success_rate.eq(1).all())

    def test_missing_real_data_fails_before_any_subprocess(self):
        config = seed_config(self.config, 42, self.directory / 'missing_real')
        config['dataset_kind'] = 'real'
        config['paths']['training_csv'] = str(self.directory / 'missing_train_A.csv')
        path = self.directory / 'missing_real_config.yaml'
        with path.open('w', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream)
        with patch('src.run_experiment.subprocess.run') as child:
            with self.assertRaisesRegex(FileNotFoundError, MISSING_DATA_MESSAGE):
                run_experiment(path)
            child.assert_not_called()
        self.assertFalse(resolve_path(config['paths']['comparison_csv']).exists())


class ReproducibilityTests(unittest.TestCase):
    def test_random_baseline_deterministic_and_unique(self):
        first = random_predictions(['a', 'b', 'c', 'd', 'e', 'f'], ['q1', 'q2'], seed=42)
        second = random_predictions(['f', 'e', 'd', 'c', 'b', 'a', 'a'], ['q2', 'q1'], seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first.duplicated(['question_id', 'player_id']).any())

    def test_coverage_failure_counts_as_zero_positions(self):
        row = coverage_row('unit', {'input_games': 2, 'valid_games': 1, 'skipped_games': 1,
                                    'records': [{'num_positions': 4}]})
        self.assertEqual(row['success_rate'], 0.5)
        self.assertEqual(row['mean_sampled_positions'], 2)
        self.assertEqual(row['min_sampled_positions'], 0)
        self.assertEqual(row['max_sampled_positions'], 4)

    def test_multi_seed_paths_and_mean_std(self):
        config = load_config(ROOT / 'configs/real_quick.yaml')
        tables = []
        for seed, score in [(42, 0.2), (123, 0.4), (2026, 0.6)]:
            settings = seed_config(config, seed, f'outputs/seed_{seed}')
            self.assertEqual(settings['paths']['training_csv'], config['paths']['training_csv'])
            for key, value in settings['paths'].items():
                if key != 'training_csv':
                    self.assertTrue(value.startswith(f'outputs/seed_{seed}/'))
            tables.append(pd.DataFrame([{'method': method, 'seed': seed, 'top1': score,
                                         'top3': score, 'top5': score, 'competition_score': score}
                                        for method in ['random', 'opening', 'triplet']]))
        summary = summarize_seeds(tables)
        self.assertEqual(len(summary), 15)
        np.testing.assert_allclose(summary[summary.seed == 'mean'].competition_score, 0.4)
        np.testing.assert_allclose(summary[summary.seed == 'std'].competition_score, 0.2)

    def test_summary_redacts_external_paths(self):
        safe = sanitized_config({'windows': 'C:\\Users\\someone\\secret.txt',
                                 'posix': '/home/someone/private', 'relative': 'data/training/train_A.csv'})
        self.assertEqual(safe['windows'], '<external-path>')
        self.assertEqual(safe['posix'], '<external-path>')
        self.assertEqual(safe['relative'], 'data/training/train_A.csv')

    def test_all_new_module_imports(self):
        for name in ['experiment_state', 'evaluate_test', 'random_baseline', 'opening_test',
                     'analyze_feature_coverage', 'experiment_summary', 'run_experiment']:
            importlib.import_module(f'src.{name}')
        importlib.import_module('scripts.run_multi_seed')


if __name__ == '__main__':
    unittest.main()
