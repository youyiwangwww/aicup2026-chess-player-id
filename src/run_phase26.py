"""Run controlled DEV ablations; never evaluate or read FINAL TEST 2 truth."""
import argparse
import copy
import json
import time

import pandas as pd
import yaml

from .analyze_embeddings import analyze_embeddings
from .analyze_opening_signal import analyze_opening
from .analyze_feature_coverage import coverage_row
from .build_metric_split import audit_split
from .experiment_state import experiment_signature, read_json
from .metric_utils import file_digest, write_json
from .phase26_common import dev_frames, dev_only, variant_config
from .phase26_cross_color import analyze_cross_color
from .phase26_prepare import prepare, preserve_round1
from .phase26_selection import freeze_selection
from .final_test2_lock import require_unconsumed
from .phase26_train import train_dev
from .player_features import preprocess_partition
from .utils import ROOT, load_config, resolve_path, write_csv


def save_config(config, name):
    """Store the exact reproducible config for each experiment."""
    path = ROOT / 'configs' / f'phase26_{name}.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')


def prepare_dev_pool(base):
    """Export only DEV VAL players for later cross-color sampling."""
    destination = ROOT / 'outputs/phase26/splits/dev_val_pool.csv'
    if destination.exists():
        return
    _, _, _, truth = dev_frames(base)
    from .utils import read_csv
    source = read_csv(base['paths']['training_csv'], ['player_id', 'game_id', 'color', 'sgf_content'])
    pool = source[source.player_id.isin(truth.player_id)][['player_id', 'game_id', 'color', 'sgf_content']]
    write_csv(pool, destination)


def common_features(base, positions):
    """Cache TRAIN/DEV VAL only; FINAL TEST 2 preprocessing is deliberately deferred."""
    config = copy.deepcopy(base)
    directory = f'outputs/phase26/features/k{positions}'
    config['features']['max_positions_per_game'] = positions
    config['paths']['cache_dir'] = directory
    jobs = [('train', 'player_id', 'metric_train_csv', 'train_manifest'),
            ('val_candidate', 'player_id', 'val_candidates_csv', 'val_candidate_manifest'),
            ('val_query', 'question_id', 'val_queries_csv', 'val_query_manifest')]
    frames = dev_frames(base)[:3]
    errors, coverage = [], []
    for frame, (name, column, source, manifest) in zip(frames, jobs):
        config['paths'][manifest] = f'{directory}/{name}.json'
        path = resolve_path(config['paths'][manifest])
        if path.exists():
            cached = read_json(path)
            if (cached['source_sha256'] != file_digest(config['paths'][source])
                    or cached['features'] != config['features'] or cached['seed'] != config['seed']):
                raise ValueError('Existing DEV cache changed; refusing silent replacement')
            skipped = read_json(f'{directory}/{name}_errors.json')
        else:
            cached, skipped = preprocess_partition(frame, column, config['paths'][source],
                                                  config['paths'][manifest], config)
            write_json(skipped, f'{directory}/{name}_errors.json')
        coverage.append(coverage_row(name, cached))
        errors.extend(skipped)
    write_csv(pd.DataFrame(coverage), f'{directory}/coverage.csv')
    write_csv(pd.DataFrame(errors, columns=['partition', 'group_id', 'game_id', 'error']), f'{directory}/errors.csv')
    return config


def prepare_variant(base, features, name, players, games, positions, epochs, strategy='random', original=False):
    """Use nested fixed TRAIN subsets while keeping DEV VAL identical in every ablation."""
    config = variant_config(base, name, players, games, positions, epochs, strategy, original)
    existing_config = ROOT / 'configs' / f'phase26_{name}.yaml'
    if existing_config.exists() and experiment_signature(load_config(existing_config)) != experiment_signature(config):
        raise ValueError(f'{name} settings changed; refuse to overwrite/reuse an existing DEV experiment')
    training, candidates, queries, truth = dev_frames(base)
    # Seeded player order and existing seeded game order yield nested identity/game subsets.
    import numpy as np
    order = np.random.default_rng(base['seed']).permutation(sorted(training.player_id.unique()))
    subset = pd.concat([training[training.player_id == player].iloc[:games] for player in order[:players]],
                       ignore_index=True)
    if subset.player_id.nunique() != players or len(subset) != players * games:
        raise ValueError('Insufficient nested TRAIN subset')
    audit_split(subset, candidates, queries, truth, base['seed'])
    write_csv(subset, config['paths']['metric_train_csv'])
    audit = copy.deepcopy(read_json(base['paths']['split_audit_json']))
    audit.update(train_player_count=players, train_game_count=len(subset))
    audit['partition_sha256']['metric_train_csv'] = file_digest(config['paths']['metric_train_csv'])
    write_json(audit, config['paths']['split_audit_json'])
    manifest = copy.deepcopy(read_json(features['paths']['train_manifest']))
    ids = set(subset.game_id)
    manifest['records'] = [record for record in manifest['records'] if record['game_id'] in ids]
    # Preserve subset record order to hold random sampling fixed across runs.
    records = {record['game_id']: record for record in manifest['records']}
    manifest['records'] = [records[game] for game in subset.game_id if game in records]
    manifest.update(source_sha256=file_digest(config['paths']['metric_train_csv']), input_games=len(subset),
                    valid_games=len(manifest['records']), skipped_games=len(subset) - len(manifest['records']))
    directory = config['output_dir']
    config['paths'].update(cache_dir=features['paths']['cache_dir'], train_manifest=f'{directory}/train_manifest.json',
        val_candidate_manifest=features['paths']['val_candidate_manifest'],
        val_query_manifest=features['paths']['val_query_manifest'])
    write_json(manifest, config['paths']['train_manifest'])
    save_config(config, name)
    return config


def result_row(config, best):
    """Report validation-only metrics with all controlled parameters."""
    return {'experiment': config['experiment'], 'train_players': config['split']['train_players'],
            'games_per_player': config['split']['train_games_per_player'],
            'epochs': config['training']['epochs'], 'positions_per_game': config['features']['max_positions_per_game'],
            'channels': config['model']['channels'], 'blocks': config['model']['num_blocks'],
            'embedding_dim': config['model']['embedding_dim'],
            'triplet_strategy': config['training'].get('triplet_strategy', 'random'),
            'val_top1': best['val_top1'], 'val_top3': best['val_top3'], 'val_top5': best['val_top5'],
            'val_score': best['val_score'], 'best_epoch': best['epoch']}


def run(base):
    """Run A/B/C/hard sequentially with choices based exclusively on fixed DEV VAL."""
    require_unconsumed()
    started = time.perf_counter()
    prepare(base)
    configurations, results, features = {}, [], {}
    with dev_only():
        prepare_dev_pool(base)
        opening = analyze_opening(base)
        for row in opening.to_dict('records'):
            results.append({'experiment': row['experiment'], 'train_players': 0, 'games_per_player': 0,
                            'epochs': 0, 'positions_per_game': 0, 'channels': 0, 'blocks': 0,
                            'embedding_dim': 0, 'triplet_strategy': 'none', **{key: row[key] for key in
                                ['val_top1', 'val_top3', 'val_top5', 'val_score']}, 'best_epoch': 0})
        def experiment(name, players, games, positions, epochs, strategy='random', original=False):
            if positions not in features:
                features[positions] = common_features(base, positions)
            config = prepare_variant(base, features[positions], name, players, games, positions, epochs, strategy, original)
            configurations[name] = config
            state_path = resolve_path(config['paths']['training_state_json'])
            if state_path.exists():
                state = read_json(state_path)
                if state['status'] != 'completed' or state['best_checkpoint_sha256'] != file_digest(config['paths']['best_checkpoint']):
                    raise ValueError('Existing experiment checkpoint/state mismatch')
                if state.get('experiment_signature', experiment_signature(config)) != experiment_signature(config):
                    raise ValueError('Completed DEV experiment settings mismatch')
                history = pd.read_csv(resolve_path(config['paths']['training_log_csv']))
                best = history.loc[history.val_score.idxmax()].to_dict()
                print(f'{name}: reused completed DEV run (no TEST)', flush=True)
            else:
                best = train_dev(config)
            result = result_row(config, best)
            results.append(result)
            write_csv(pd.DataFrame(results), 'outputs/phase26/dev_results.csv')
            return result
        reference = experiment('Phase25-reference', 30, 10, 4, 3, original=True)
        a_results = []
        for index, (players, games) in enumerate(base['diagnosis']['data_sizes'], 1):
            a_results.append(experiment(f'A{index}', players, games, 8, 10))
        best_a = max(a_results, key=lambda row: row['val_score'])
        # 8 positions is already trained in A: reuse it rather than rerunning for selection.
        b_results = []
        for positions in base['diagnosis']['positions']:
            if positions == 8:
                reused = dict(best_a, experiment='B8')
                configurations['B8'] = configurations[best_a['experiment']]
                results.append(reused)
                b_results.append(reused)
                write_csv(pd.DataFrame(results), 'outputs/phase26/dev_results.csv')
            else:
                b_results.append(experiment(f'B{positions}', best_a['train_players'], best_a['games_per_player'], positions, 10))
        best_b = max(b_results, key=lambda row: row['val_score'])
        experiment('C20-Random', best_b['train_players'], best_b['games_per_player'],
                   best_b['positions_per_game'], base['diagnosis']['max_epochs'])
        experiment('H20-Hard', best_b['train_players'], best_b['games_per_player'],
                   best_b['positions_per_game'], base['diagnosis']['max_epochs'], strategy='batch_hard')
        triplets = [row for row in results if row['triplet_strategy'] in ['random', 'batch_hard']]
        selected = max(triplets, key=lambda row: row['val_score'])
        chosen = copy.deepcopy(configurations[selected['experiment']])
        chosen['selection'] = {'criterion': 'DEV VAL competition score only', 'selected_experiment': selected['experiment'],
            'best_epoch': int(selected['best_epoch']), 'dev_val_score': selected['val_score'],
            'reference_dev_val_score': reference['val_score'],
            'score_improvement_same_dev': selected['val_score'] - reference['val_score'],
            'reason': 'Highest DEV VAL score among prespecified triplet experiments; ties keep the first.',
            'final_test2_status': 'LOCKED; never used for tuning or evaluated',
            'round1_comparison': 'Round 1 TEST score is historical only; a different 50-player DEV is not directly comparable.'}
        save_config(chosen, 'selected')
        write_json(chosen['selection'], 'outputs/phase26/selection.json')
        analyze_cross_color(base, chosen)
        analyze_embeddings(chosen)
        freeze_selection()
        write_json({'status': 'completed', 'elapsed_seconds': time.perf_counter() - started,
                    'selection': chosen['selection'], 'final_test2_evaluations': 0}, 'outputs/phase26/summary.json')
    preserve_round1(base)
    print('Phase 2.6 completed. FINAL TEST 2 remains LOCKED and unevaluated.', flush=True)


def main():
    """Start/resume DEV experiments without offering a final-test execution flag."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/phase26.yaml'))
    run(load_config(parser.parse_args().config))


if __name__ == '__main__':
    main()
