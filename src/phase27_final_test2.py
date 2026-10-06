"""One authorized evaluation of frozen FINAL TEST 2, with irreversible closure."""
import argparse
import copy
import time
from datetime import datetime, timezone

import pandas as pd
import torch

from .baseline import predict
from .embed_players import load_encoder
from .evaluate import evaluate
from .experiment_state import read_json
from .final_test2_lock import claim_once, final_directory
from .metric_utils import choose_device, file_digest, seed_everything, write_json
from .opening_features import build_fingerprints
from .phase26_common import MemoryStore, dev_retrieve
from .phase26_prepare import preserve_round1
from .phase27_preflight import preflight, require
from .player_features import preprocess_partition
from .random_baseline import random_predictions
from .utils import ROOT, write_csv


def verify_unchanged(config, safety):
    """Reject source/config/checkpoint mutations before committing any scores."""
    require(file_digest(ROOT / 'configs/phase26_selected.yaml') == safety['config_sha256'], 'Config changed during evaluation')
    require(file_digest(config['paths']['best_checkpoint']) == safety['checkpoint_sha256'], 'Checkpoint changed during evaluation')
    require(file_digest(config['paths']['training_csv']) == safety['dataset_sha256'], 'Dataset changed during evaluation')
    mapping = {'metric_train_csv': 'outputs/phase26/splits/dev_train.csv',
               **{key: config['paths'][key] for key in safety['split_sha256'] if key != 'metric_train_csv'}}
    for key, digest in safety['split_sha256'].items():
        require(file_digest(mapping[key]) == digest, f'Split changed during evaluation: {key}')
    preserve_round1(config)


def run_once(config_path=None):
    """Preflight, preprocess, infer all three fixed methods, score once, then close."""
    started = time.perf_counter()
    config, candidates, queries, truth, safety = preflight(config_path)
    directory = final_directory()
    with claim_once(directory) as started_at:
        write_json(safety, directory / 'safety_check.json')
        runtime = copy.deepcopy(config)
        runtime['paths'].update(cache_dir=str(directory / 'features'),
                                test_candidate_manifest=str(directory / 'features/candidate.json'),
                                test_query_manifest=str(directory / 'features/query.json'))
        coverage, errors = [], []
        for label, frame, group, source_key, manifest_key in [
            ('candidate', candidates, 'player_id', 'test_candidates_csv', 'test_candidate_manifest'),
            ('query', queries, 'question_id', 'test_queries_csv', 'test_query_manifest')]:
            manifest, failed = preprocess_partition(frame, group, runtime['paths'][source_key],
                                                    runtime['paths'][manifest_key], runtime)
            coverage.append({'partition': label, 'total_games': len(frame),
                'successful_games': manifest['valid_games'], 'failed_games': manifest['skipped_games'],
                'success_rate': manifest['valid_games'] / len(frame)})
            errors.extend(failed)
        write_csv(pd.DataFrame(coverage), directory / 'feature_coverage.csv')
        write_csv(pd.DataFrame(errors, columns=['partition', 'group_id', 'game_id', 'error']), directory / 'feature_errors.csv')
        predictions = {'random': random_predictions(candidates.player_id, queries.question_id, seed=42, top_k=5)}
        cf, ce = build_fingerprints(candidates, 'player_id', opening_moves=10, board_size=19)
        qf, qe = build_fingerprints(queries, 'question_id', opening_moves=10, board_size=19)
        require(set(cf) == set(candidates.player_id), 'Opening cannot drop candidate identities')
        write_csv(pd.concat([ce, qe], ignore_index=True), directory / 'opening10_errors.csv')
        predictions['opening10'] = predict(cf, qf, 5)
        seed_everything(config['seed'], config['training']['deterministic'], config['training']['num_threads'])
        device = choose_device(config['training']['device'])
        require(file_digest(config['paths']['best_checkpoint']) == safety['checkpoint_sha256'], 'Checkpoint changed before inference')
        model, checkpoint = load_encoder(config['paths']['best_checkpoint'], config, device)
        require(checkpoint['epoch'] == 17 and checkpoint['model_config'] == config['model'], 'Loaded checkpoint mismatch')
        model.eval()
        model.requires_grad_(False)
        cs, qs = MemoryStore(runtime, 'test_candidate'), MemoryStore(runtime, 'test_query')
        require({r['group_id'] for r in cs.records} == set(candidates.player_id), 'Features cannot drop candidate identities')
        print(f'Triplet inference only: device={device}, checkpoint epoch=17, optimizer/backprop disabled', flush=True)
        predictions['triplet_hard'], _, _ = dev_retrieve(model, cs, qs, device, config['inference']['batch_size'])
        verify_unchanged(config, safety)
        metrics, rows = {}, []
        names = {'random': 'random_predictions.csv', 'opening10': 'opening10_predictions.csv',
                 'triplet_hard': 'triplet_predictions.csv'}
        for method, rankings in predictions.items():
            write_csv(rankings, directory / names[method])
            values, details = evaluate(rankings, truth)
            values = {key: float(value) for key, value in values.items()}
            metrics[method] = values
            write_csv(details, directory / f'{method}_evaluation.csv')
            rows.append({'method': method, 'top1': values['top_1_accuracy'], 'top3': values['top_3_accuracy'],
                         'top5': values['top_5_accuracy'], 'competition_score': values['competition_score']})
        comparison = pd.DataFrame(rows, columns=['method', 'top1', 'top3', 'top5', 'competition_score'])
        write_csv(comparison, directory / 'final_comparison.csv')
        receipt = {**safety, 'status': 'CLOSED TEST', 'evaluation_count': 1, 'timestamp': datetime.now(timezone.utc).isoformat(),
            'started_at': started_at, 'metrics': metrics, 'feature_coverage': coverage,
            'seed': 42, 'opening_total_moves': 10, 'device': str(device), 'torch_version': str(torch.__version__),
            'elapsed_seconds': time.perf_counter() - started, 'training_performed': False}
        write_json(receipt, directory / 'final_test2_receipt.json')
        (directory / 'CLOSED_TEST').write_text('FINAL TEST 2 CLOSED TEST; evaluation_count=1; re-evaluation forbidden.\n', encoding='utf-8')
        write_json({'status': 'CLOSED TEST', 'evaluation_count': 1, 'timestamp': receipt['timestamp']}, directory / 'one_shot_attempt.json')
        summary = read_json('outputs/phase26/summary.json')
        summary.update(final_test2_evaluations=1, final_test2_status='CLOSED TEST',
                       final_test2_receipt='outputs/phase26/final_test2/final_test2_receipt.json')
        write_json(summary, 'outputs/phase26/summary.json')
        print('FINAL TEST 2 — ONE-SHOT RESULT', flush=True)
        print(comparison.to_string(index=False), flush=True)
        print('FINAL TEST 2 permanently CLOSED TEST; future invocations are rejected.', flush=True)
        return receipt


def main():
    """Expose read-only preflight and the single final evaluation, with no tuning flags."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.check_only:
            _, _, _, _, report = preflight()
            print(f'SAFETY CHECK PASSED: 50 players, 2000 candidates, 500 queries; evaluation_count=0; checkpoint {report["checkpoint_sha256"]}')
        else:
            run_once()
    except (ValueError, PermissionError, FileNotFoundError) as exc:
        parser.exit(2, f'REFUSED / STOPPED: {exc}\n')


if __name__ == '__main__':
    main()
