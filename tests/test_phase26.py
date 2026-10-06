"""DEV isolation, hard-negative correctness and game aggregation regression tests."""
import builtins
import copy
import hashlib
import importlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from src.analyze_embeddings import analyze_embeddings, similarity_statistics
from src.analyze_opening_signal import analyze_opening, opening_metrics
from src.build_metric_split import THREE_WAY_KEYS, build_three_way_split
from src.create_metric_mock_data import metric_mock_frame
from src.embed_players import embed_groups
from src.metric_utils import file_digest, write_json
from src.phase26_common import MemoryStore, aggregate_games, dev_only, embed_games
from src.phase26_cross_color import cross_color_partitions
from src.phase26_train import nearest_other_player, train_dev
from src.player_features import preprocess_partition
from src.player_model import PlayerEncoder
from src.run_phase26 import prepare_variant
from src.utils import ROOT, load_config, write_csv


class DiagnosisTests(unittest.TestCase):
    def test_changed_experiment_settings_rejected_before_data_writes(self):
        """A resume must reject changed hyperparameters before opening or rewriting DEV CSVs."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'configs').mkdir()
            base = load_config(ROOT / 'configs/phase26.yaml')
            from src.phase26_common import variant_config
            previous = variant_config(base, 'A1', 30, 10, 8, 10)
            previous['model']['channels'] = 32
            (directory / 'configs/phase26_A1.yaml').write_text(yaml.safe_dump(previous), encoding='utf-8')
            with patch('src.run_phase26.ROOT', directory), patch('src.run_phase26.dev_frames') as reader:
                with self.assertRaisesRegex(ValueError, 'settings changed'):
                    prepare_variant(base, {}, 'A1', 30, 10, 8, 10)
            reader.assert_not_called()

    def test_all_phase26_imports(self):
        for module in ['phase26_common', 'phase26_prepare', 'phase26_train', 'run_phase26',
                       'analyze_opening_signal', 'analyze_embeddings', 'phase26_cross_color', 'phase26_selection']:
            importlib.import_module('src.' + module)

    def test_in_memory_csv_digest_matches_written_bytes(self):
        """Split metadata can hash generated truth without reopening a locked file."""
        frame = pd.DataFrame([{'question_id': 'q', 'player_id': '測試'}])
        expected = hashlib.sha256(frame.to_csv(index=False).encode('utf-8-sig')).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'truth.csv'
            write_csv(frame, path)
            self.assertEqual(file_digest(path), expected)

    def test_guard_blocks_relative_closed_test_paths(self):
        with dev_only():
            for name in ['outputs/splits/test_candidates.csv', 'outputs/round1_archive/outputs/results/test_metrics.csv']:
                with self.assertRaises(PermissionError):
                    open(name)

    def test_final_truth_guard_covers_actual_open_apis(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'final_test2_ground_truth.csv'
            target.write_text('question_id,player_id\nq,p\n', encoding='utf-8')
            with dev_only():
                for operation in [lambda: builtins.open(target), lambda: io.open(target),
                                  lambda: target.open(), lambda: pd.read_csv(target),
                                  lambda: os.open(target, os.O_RDONLY)]:
                    with self.assertRaises(PermissionError):
                        operation()
            # Guard scope ends normally; preparation may write/read its own generated artifacts.
            self.assertIn('player_id', target.read_text())

    def test_hard_negative_masks_same_player_and_retains_gradient(self):
        anchors = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        pool = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]], requires_grad=True)
        negatives, indices = nearest_other_player(anchors, pool, torch.tensor([0, 1]), torch.tensor([0, 0, 1, 1]))
        self.assertEqual(indices.tolist(), [3, 1])
        negatives.sum().backward()
        self.assertEqual(pool.grad.sum().item(), 4.)
        with self.assertRaises(ValueError):
            nearest_other_player(anchors, pool, torch.tensor([0, 0]), torch.zeros(4, dtype=torch.long))

    def test_batched_embedding_preserves_equal_game_weights(self):
        class Store:
            records = [{'group_id': 'a'}, {'group_id': 'a'}, {'group_id': 'b'}]
            arrays = [np.random.default_rng(index + 42).integers(0, 2, (count, 17, 19, 19), dtype=np.uint8)
                      for index, count in enumerate([1, 3, 2])]
            def __len__(self):
                return len(self.records)
            def features(self, index):
                return self.arrays[index]
        torch.manual_seed(42)
        model = PlayerEncoder(channels=4, num_blocks=1, embedding_dim=4)
        store = Store()
        actual = aggregate_games(embed_games(model, store, torch.device('cpu'), 4), store.records)
        expected = embed_groups(model, store, torch.device('cpu'), 4)
        self.assertGreater(np.linalg.norm(expected['a'] - expected['b']), 1e-5)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-6)

    def test_similarity_statistics_use_identity_not_color(self):
        c = np.array([[1., 0.], [0., 1.]])
        frame, summary = similarity_statistics(c, c, ['a', 'b'], ['a', 'b'])
        self.assertEqual(frame.pair_count.tolist(), [2, 2])
        self.assertEqual(frame['mean'].tolist(), [1., 0.])
        self.assertAlmostEqual(summary['separation'], 1.)

    def test_cross_color_fixed_counts_disjoint_and_reproducible(self):
        pool = metric_mock_frame(4, 8)
        first, audit = cross_color_partitions(pool, 3, 2, 42)
        repeat, _ = cross_color_partitions(pool.sample(frac=1, random_state=11), 3, 2, 42)
        self.assertEqual(audit['eligible_players'], 4)
        for actual, expected in zip(first, repeat):
            _, candidates, queries, truth = actual
            self.assertEqual(len(candidates), 12)
            self.assertEqual(len(queries), 8)
            self.assertFalse(set(candidates.game_id) & set(queries.game_id))
            self.assertNotIn('player_id', queries.columns)
            pd.testing.assert_frame_equal(candidates, expected[1])
            pd.testing.assert_frame_equal(queries, expected[2])


class DevTuningIntegrationTests(unittest.TestCase):
    def test_tuning_entry_points_never_open_final_truth(self):
        """Run real hard training/opening/embedding code with FINAL truth access forbidden."""
        (ROOT / '.test-tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.test-tmp') as directory:
            directory = Path(directory)
            config = load_config(ROOT / 'configs/phase26.yaml')
            config['split'] = load_config(ROOT / 'configs/phase25_mock.yaml')['split']
            config['experiment'] = 'test-hard'
            config['features']['max_positions_per_game'] = 2
            config['model'].update(channels=4, num_blocks=1, embedding_dim=4)
            config['training'].update(epochs=2, samples_per_epoch=8, batch_size=4, num_threads=1,
                                      triplet_strategy='batch_hard')
            parts = build_three_way_split(metric_mock_frame(14, 6), **config['split'], seed=42)
            for key, frame in zip(THREE_WAY_KEYS, parts[:7]):
                filename = 'final_test2_ground_truth.csv' if key == 'test_ground_truth_csv' else f'{key}.csv'
                config['paths'][key] = str(directory / filename)
                write_csv(frame, config['paths'][key])
            audit = parts[-1]
            audit['partition_sha256'] = {key: file_digest(config['paths'][key]) for key in THREE_WAY_KEYS}
            config['paths']['split_audit_json'] = str(directory / 'audit.json')
            write_json(audit, config['paths']['split_audit_json'])
            config['paths']['cache_dir'] = str(directory / 'features')
            for key in ['best_checkpoint', 'last_checkpoint', 'training_state_json', 'training_log_csv']:
                config['paths'][key] = str(directory / key)
            for frame, group, key, manifest in [(parts[0], 'player_id', 'metric_train_csv', 'train_manifest'),
                    (parts[1], 'player_id', 'val_candidates_csv', 'val_candidate_manifest'),
                    (parts[2], 'question_id', 'val_queries_csv', 'val_query_manifest')]:
                config['paths'][manifest] = str(directory / f'{manifest}.json')
                preprocess_partition(frame, group, config['paths'][key], config['paths'][manifest], config)
            # Capture actual opens, not only mocked function calls or source-code scanning.
            opened = []
            real_open = io.open
            def capture(path, *args, **kwargs):
                opened.append(str(path))
                if 'final_test2_ground_truth' in str(path):
                    raise AssertionError('Tuning tried to open FINAL TEST 2 truth')
                return real_open(path, *args, **kwargs)
            with patch('io.open', side_effect=capture):
                best = train_dev(config)
                with patch('src.analyze_opening_signal.write_csv'):
                    opening = analyze_opening(config)
                with patch('src.analyze_embeddings.write_csv'), patch('src.analyze_embeddings.write_json'):
                    statistics, summary = analyze_embeddings(config)
            self.assertTrue(any('val_ground_truth_csv' in name for name in opened))
            self.assertFalse(any('final_test2_ground_truth' in name for name in opened))
            self.assertEqual(len(opening), 6)
            self.assertEqual(len(statistics), 2)
            self.assertIn(best['epoch'], [1, 2])


if __name__ == '__main__':
    unittest.main()
