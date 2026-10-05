"""Correctness and small real training tests for the Go metric-learning pipeline."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.build_metric_split import audit_split, build_metric_split
from src.compare_baselines import compare
from src.create_metric_mock_data import metric_mock_frame
from src.embed_players import embed_groups, load_encoder, retrieve
from src.evaluate import evaluate
from src.feature_cache import GameFeatureStore, load_store
from src.metric_utils import file_digest, seed_everything, write_json
from src.player_features import extract_player_features, preprocess_partition
from src.player_model import PlayerEncoder
from src.train_triplet import train
from src.triplet_dataset import TripletDataset
from src.utils import ROOT, load_config, write_csv


class PlayerFeatureTests(unittest.TestCase):
    def test_black_target_pre_move_and_shape(self):
        features = extract_player_features('(;SZ[19];B[aa];W[bb];B[cc])', 'B')
        self.assertEqual(features.shape, (2, 17, 19, 19))
        self.assertEqual(features.dtype, np.uint8)
        self.assertEqual(features[0, :16].sum(), 0)
        self.assertTrue(features[:, 16].all())
        self.assertEqual(features[1, 7, 18, 0], 1)  # existing B[aa]
        self.assertEqual(features[1, 15, 17, 1], 1)  # existing W[bb]
        self.assertEqual(features[1, 7, 16, 2], 0)  # B[cc] not played yet

    def test_white_target_pre_move(self):
        features = extract_player_features('(;SZ[19];B[aa];W[bb];B[cc];W[dd])', 'W')
        self.assertEqual(features.shape, (2, 17, 19, 19))
        self.assertEqual(features[:, 16].sum(), 0)
        self.assertEqual(features[0, :8].sum(), 0)  # no white stones yet
        self.assertEqual(features[0, 15, 18, 0], 1)
        self.assertEqual(features[1, 7, 17, 1], 1)
        self.assertEqual(features[1, 15, 16, 2], 1)
        self.assertEqual(features[1, 7, 15, 3], 0)

    def test_history_order_and_empty_padding(self):
        features = extract_player_features('(;SZ[19];B[aa];W[bb];B[cc])', 'B')
        self.assertEqual(features[1, :5].sum(), 0)
        self.assertEqual(features[1, 5].sum(), 0)  # initial empty board
        self.assertEqual(features[1, 6, 18, 0], 1)  # board before white turn
        self.assertEqual(features[1, 14].sum(), 0)  # white not placed in previous state

    def test_setup_and_capture(self):
        content = '(;SZ[19]AB[ab][ba][cb]AW[bb];B[bc];W[ss];B[dd])'
        features = extract_player_features(content, 'B')
        self.assertEqual(features[0, 7].sum(), 3)
        self.assertEqual(features[0, 15, 17, 1], 1)
        self.assertEqual(features[1, 15, 17, 1], 0)  # white bb was captured
        self.assertEqual(features[1, 15, 0, 18], 1)  # white ss present
        self.assertEqual(features[1, 7, 16, 1], 1)  # black bc present
        self.assertEqual(features[1, 7, 15, 3], 0)  # black dd still absent

    def test_pass_and_main_variation(self):
        features = extract_player_features('(;SZ[19];B[];W[aa];B[bb];W[];B[cc])', 'B')
        self.assertEqual(len(features), 3)
        self.assertEqual(features[1, 7].sum(), 0)
        self.assertEqual(features[2, 7, 17, 1], 1)
        self.assertEqual(features[2, 15, 18, 0], 1)
        main = extract_player_features('(;SZ[19];B[aa](;W[bb];B[cc])(;W[ss];B[rr]))', 'W')
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0, 15, 18, 0], 1)

    def test_sampling_is_bounded_and_reproducible(self):
        content = metric_mock_frame().iloc[0].sgf_content
        first = extract_player_features(content, 'B', max_positions_per_game=4, seed=123)
        second = extract_player_features(content, 'B', max_positions_per_game=4, seed=123)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 4)

    def test_invalid_games_raise_per_game_errors(self):
        for content, color in [('bad', 'B'), ('', 'B'), ('(;SZ[9];B[aa])', 'B'),
                               ('(;SZ[19];B[aa];W[aa])', 'B'), ('(;SZ[19];W[aa])', 'B'),
                               ('(;SZ[19];B[aa])', '')]:
            with self.assertRaises(ValueError):
                extract_player_features(content, color)


class MetricSplitTests(unittest.TestCase):
    def test_disjoint_ids_games_sgfs_and_reproducibility(self):
        frame = metric_mock_frame()
        split = build_metric_split(frame, 4, 6, 4, 3, 2, 42)
        repeat = build_metric_split(frame.sample(frac=1, random_state=8), 4, 6, 4, 3, 2, 42)
        for left, right in zip(split[:4], repeat[:4]):
            pd.testing.assert_frame_equal(left, right)
        training, candidates, queries, truth, audit = split
        self.assertFalse(set(training.player_id) & set(candidates.player_id))
        self.assertNotIn('player_id', queries)
        self.assertNotIn('rank', training)
        self.assertNotIn('rank', queries)
        self.assertEqual(audit['training_game_count'], 16)
        self.assertEqual(audit['candidate_game_count'], 18)
        self.assertEqual(audit['query_game_count'], 12)
        self.assertTrue(all(count == 0 for count in audit['overlap_counts'].values()))
        self.assertEqual(candidates.groupby('player_id').size().unique().tolist(), [3])
        self.assertEqual(queries.groupby('question_id').size().unique().tolist(), [2])
        self.assertEqual(len(truth), 6)

    def test_all_cross_partition_leaks_raise(self):
        training, candidates, queries, truth, _ = build_metric_split(metric_mock_frame(), 4, 6, 4, 3, 2)
        for first, second in [(training, candidates), (training, queries), (candidates, queries)]:
            for column in ['game_id', 'sgf_content']:
                corrupted = second.copy()
                corrupted.loc[0, column] = first.loc[0, column]
                parts = [training, candidates, queries]
                parts[next(i for i, frame in enumerate(parts) if frame is second)] = corrupted
                with self.assertRaisesRegex(ValueError, 'leakage'):
                    audit_split(*parts, truth, 42)
        corrupted_train = training.copy()
        corrupted_train.loc[0, 'player_id'] = truth.loc[0, 'player_id']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            audit_split(corrupted_train, candidates, queries, truth, 42)

    def test_insufficient_players_not_silently_reduced(self):
        with self.assertRaisesRegex(ValueError, 'Need 10 evaluation players'):
            build_metric_split(metric_mock_frame(), 30, 10, 10, 10, 5)
        with self.assertRaisesRegex(ValueError, 'disjoint identities'):
            build_metric_split(metric_mock_frame(), 5, 6, 4, 3, 2)


class CachedMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build a tiny independent on-disk split/cache used only by these tests."""
        (ROOT / '.test-tmp').mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / '.test-tmp')
        cls.directory = Path(cls.temp.name)
        cls.config = load_config(ROOT / 'configs/triplet_mock.yaml')
        for key, value in cls.config['paths'].items():
            relative = Path(value).relative_to('data/mock' if key == 'training_csv' else 'outputs/phase2_mock')
            cls.config['paths'][key] = str(cls.directory / relative)
        cls.config['training'].update(samples_per_epoch=8, num_threads=1)
        cls.config['model'].update(channels=8, num_blocks=1, embedding_dim=8)
        frame = metric_mock_frame()
        partitions = build_metric_split(frame, **cls.config['split'], seed=cls.config['seed'])
        for key, partition in zip(['metric_train_csv', 'candidates_csv', 'queries_csv', 'ground_truth_csv'], partitions):
            write_csv(partition, cls.config['paths'][key])
        for partition, column, csv_key, manifest_key in [
            (partitions[0], 'player_id', 'metric_train_csv', 'train_manifest'),
            (partitions[1], 'player_id', 'candidates_csv', 'candidate_manifest'),
            (partitions[2], 'question_id', 'queries_csv', 'query_manifest'),
        ]:
            preprocess_partition(partition, column, cls.config['paths'][csv_key],
                                 cls.config['paths'][manifest_key], cls.config)
        cls.truth = partitions[3]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_triplet_constraints_reproducibility_and_epoch(self):
        store = load_store(self.config, 'train')
        dataset = TripletDataset(store, samples_per_epoch=32, seed=42)
        repeat = TripletDataset(store, samples_per_epoch=32, seed=42)
        for index in range(32):
            self.assertEqual(dataset.sample_indices(index), repeat.sample_indices(index))
            (a, _), (p, _), (n, _) = dataset.sample_indices(index)
            self.assertEqual(store.records[a]['group_id'], store.records[p]['group_id'])
            self.assertNotEqual(store.records[a]['game_id'], store.records[p]['game_id'])
            self.assertNotEqual(store.records[a]['group_id'], store.records[n]['group_id'])
        tensors = dataset[0]
        self.assertEqual(tensors[0].shape, (17, 19, 19))
        self.assertEqual(tensors[0].dtype, torch.float32)
        dataset.set_epoch(1)
        repeat.set_epoch(1)
        self.assertEqual(dataset.sample_indices(2), repeat.sample_indices(2))
        self.assertTrue(any(dataset.sample_indices(i) != TripletDataset(store, 32, 42).sample_indices(i)
                            for i in range(10)))

    def test_model_output_and_unit_norm(self):
        seed_everything(42, num_threads=1)
        model = PlayerEncoder(**self.config['model']).eval()
        with torch.inference_mode():
            output = model(torch.rand(3, 17, 19, 19))
        self.assertEqual(output.shape, (3, 8))
        torch.testing.assert_close(output.norm(p=2, dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 16, 19, 19))

    def test_cache_matches_config_and_is_bounded(self):
        altered = copy.deepcopy(self.config)
        altered['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'Stale'):
            load_store(altered, 'train')
        modified_csv = self.directory / 'modified_train.csv'
        with Path(self.config['paths']['metric_train_csv']).open('rb') as stream:
            modified_csv.write_bytes(stream.read() + b'\n')
        altered = copy.deepcopy(self.config)
        altered['paths']['metric_train_csv'] = str(modified_csv)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            load_store(altered, 'train')
        store = GameFeatureStore(self.config['paths']['candidate_manifest'], max_cached_shards=1)
        for index in range(len(store)):
            self.assertEqual(store.features(index).shape[1:], (17, 19, 19))
            self.assertLessEqual(len(store._cache), 1)

    def test_preprocessing_skips_invalid_game(self):
        source = self.directory / 'broken.csv'
        frame = metric_mock_frame(2, 2).iloc[:1].copy()
        frame = pd.concat([frame, frame.assign(game_id='broken', sgf_content='bad')], ignore_index=True)
        write_csv(frame, source)
        manifest, errors = preprocess_partition(frame, 'player_id', source,
                                                self.directory / 'broken.json', self.config)
        self.assertEqual(manifest['valid_games'], 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['game_id'], 'broken')

    def test_training_checkpoints_and_retrieval(self):
        seed_everything(self.config['seed'], num_threads=1)
        initial = PlayerEncoder(**self.config['model']).stem[0].weight.detach().clone()
        history = train(self.config)
        model, best = load_encoder(self.config['paths']['best_checkpoint'], self.config, torch.device('cpu'))
        last = torch.load(self.config['paths']['last_checkpoint'], weights_only=True)
        self.assertEqual(last['epoch'], 2)
        scores = [row['val_competition_score'] for row in history]
        self.assertEqual(best['epoch'], scores.index(max(scores)) + 1)
        self.assertEqual(best['best_score'], max(scores))
        self.assertFalse(torch.equal(initial, last['model_state_dict']['stem.0.weight']))
        predictions, candidates, queries = retrieve(model, load_store(self.config, 'candidate'),
                                                    load_store(self.config, 'query'), torch.device('cpu'))
        for _, group in predictions.groupby('question_id'):
            self.assertEqual(len(group), 5)
            self.assertEqual(group.player_id.nunique(), 5)
        self.assertEqual(len(queries), 6)
        for embedding in candidates.values():
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0)
        metrics, _ = evaluate(predictions, self.truth)
        self.assertTrue(0 <= metrics['competition_score'] <= 1)
        prediction_path = Path(self.config['paths']['predictions_csv'])
        write_csv(predictions, prediction_path)
        write_json({'prediction_sha256': file_digest(prediction_path),
                    'candidate_source_sha256': file_digest(self.config['paths']['candidates_csv']),
                    'query_source_sha256': file_digest(self.config['paths']['queries_csv'])},
                   prediction_path.with_suffix('.metadata.json'))
        result = compare(self.config)
        self.assertEqual(result.method.tolist(), ['opening', 'triplet'])
        prediction_path.write_bytes(prediction_path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Stale Triplet predictions'):
            compare(self.config)
        wrong = copy.deepcopy(self.config)
        wrong['features']['history_length'] = 7
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            load_encoder(self.config['paths']['best_checkpoint'], wrong, torch.device('cpu'))

    def test_all_phase2_module_imports(self):
        for name in ['metric_utils', 'build_metric_split', 'player_features', 'feature_cache',
                     'player_model', 'triplet_dataset', 'train_triplet', 'embed_players',
                     'compare_baselines', 'create_metric_mock_data']:
            importlib.import_module(f'src.{name}')


class AggregationTests(unittest.TestCase):
    def test_equal_weight_per_game(self):
        class FakeModel(nn.Module):
            def forward(self, x):
                return F.normalize(x, p=2, dim=1)

        class FakeStore:
            records = [{'group_id': 'p'}, {'group_id': 'p'}]

            def features(self, index):
                return [np.array([[1, 0], [0, 1]], dtype=np.uint8),
                        np.array([[1, 0]], dtype=np.uint8)][index]

        embedded = embed_groups(FakeModel(), FakeStore(), torch.device('cpu'), batch_size=1)['p']
        expected = np.array([0.75, 0.25])
        np.testing.assert_allclose(embedded, expected / np.linalg.norm(expected))


if __name__ == '__main__':
    unittest.main()
