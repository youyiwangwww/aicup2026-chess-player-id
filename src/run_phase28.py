"""Run fixed DEV2 comparisons without accessing any closed TEST input."""
import argparse
import copy
import json
import sys
import traceback
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml

from .feature_cache import load_store
from .metric_utils import choose_device, file_digest, write_json
from .phase26_common import MemoryStore
from .phase28_common import dev2_only, encoder, fuse, metrics, rankings
from .phase28_opening import METHODS, opening_scores
from .phase28_split import prepare_dev2, color_cohort
from .phase28_triplet import ViewStore, train_experiment, score_embeddings
from .player_features import preprocess_partition
from .utils import ROOT, load_config, read_csv, write_csv


def cached_store(config, partition, frame, column, source_key, manifest_key):
    """Reuse only source/feature/seed-verified caches; report failures explicitly."""
    manifest_path = ROOT / config['paths'][manifest_key]
    if manifest_path.exists():
        checked = load_store(config, partition)
        manifest = checked.manifest
    else:
        manifest, errors = preprocess_partition(frame, column, config['paths'][source_key],
                                                config['paths'][manifest_key], config)
        write_csv(pd.DataFrame(errors, columns=['partition', 'group_id', 'game_id', 'error']),
                  manifest_path.with_name(manifest_path.stem + '_errors.csv'))
    store = MemoryStore(config, partition)
    positions = np.array([r['num_positions'] for r in store.records])
    coverage = {'partition': partition, 'total_games': len(frame), 'successful_games': len(store),
                'failed_games': manifest['skipped_games'], 'success_rate': len(store)/len(frame),
                'mean_positions': float(positions.mean()), 'median_positions': float(np.median(positions)),
                'min_positions': int(positions.min()), 'max_positions': int(positions.max())}
    return store, coverage


def opening_runs(directory, candidates, queries, truth):
    """Compare the six predetermined opening definitions on one complete cohort."""
    rows, results = [], {}
    for name, mode, window, color_aware in METHODS:
        scores, players, questions, errors, missing = opening_scores(candidates, queries, mode, window, color_aware)
        values, _ = metrics(scores, players, questions, truth)
        rows.append({'method': name, **values, 'mode': mode, 'window': window,
                     'color_aware': color_aware, 'failed_games': len(errors), 'missing_candidate_colors': missing})
        results[name] = (scores, players, questions, mode, window, color_aware)
        np.savez_compressed(directory / (name + '_scores.npz'), scores=scores,
                            players=np.array(players), questions=np.array(questions))
        write_csv(rankings(scores, players, questions), directory / (name + '_predictions.csv'))
        write_csv(pd.DataFrame(errors), directory / (name + '_errors.csv'))
        print(f'{name}: DEV2 score={values["competition_score"]:.6f}', flush=True)
    write_csv(pd.DataFrame(rows), directory / 'opening_results.csv')
    return rows, results


def matched_color_runs(config, pool, opening_definition, model_paths):
    """Hold identities and game budgets fixed across all five color conditions."""
    controls, ids = color_cohort(pool, config['split']['candidate_games'], config['split']['query_games'], config['seed'])
    directory = ROOT / config['output_dir'] / 'color_controls'
    directory.mkdir(parents=True, exist_ok=True)
    audit = {'players': len(ids), 'player_ids': ids, 'main_val_players': config['split']['val_players'],
             'candidate_games_per_player': config['split']['candidate_games'],
             'query_games_per_player': config['split']['query_games'],
             'eligibility': 'at least 40 Black AND 40 White source games; identical IDs across five conditions',
             'reduced_cohort': len(ids) < config['split']['val_players'], 'conditions': {}}
    rows, coverage = [], []
    device = choose_device(config['training']['device'])
    for condition, candidates, queries, truth in controls:
        target = directory / condition
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in [('candidates', candidates), ('queries', queries), ('ground_truth', truth)]:
            write_csv(frame, target / (name + '.csv'))
        audit['conditions'][condition] = {'players': len(truth), 'candidate_games': len(candidates),
            'query_games': len(queries), 'game_id_overlap': len(set(candidates.game_id) & set(queries.game_id)),
            'sgf_content_overlap': len(set(candidates.sgf_content) & set(queries.sgf_content)),
            'candidate_sha256': file_digest(target / 'candidates.csv'), 'query_sha256': file_digest(target / 'queries.csv')}
        name, mode, window, _ = opening_definition
        scores, players, questions, errors, _ = opening_scores(candidates, queries, mode, window, False)
        values, _ = metrics(scores, players, questions, truth)
        rows.append({'condition': condition, 'method': name, 'players': len(ids), **values,
                     'candidate_games_per_player': len(candidates)//len(ids), 'query_games_per_player': len(queries)//len(ids)})
        # Cross-color controls require a color-unconditioned opening: a matching-color
        # candidate bank is absent by design in B->W and W->B.
        control_config = copy.deepcopy(config)
        relative = target.relative_to(ROOT).as_posix()
        control_config['paths'].update(val_candidates_csv=relative+'/candidates.csv', val_queries_csv=relative+'/queries.csv',
            val_candidate_manifest=relative+'/features/candidate.json', val_query_manifest=relative+'/features/query.json',
            cache_dir=relative+'/features')
        cs, cov = cached_store(control_config, 'val_candidate', candidates, 'player_id', 'val_candidates_csv', 'val_candidate_manifest')
        coverage.append({'condition': condition, **cov})
        qs, cov = cached_store(control_config, 'val_query', queries, 'question_id', 'val_queries_csv', 'val_query_manifest')
        coverage.append({'condition': condition, **cov})
        for model_name, checkpoint_path in model_paths.items():
            model = encoder(config, device)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            scores, _, _ = score_embeddings(model, cs, qs, device, config['inference']['batch_size'], players, questions)
            values, _ = metrics(scores, players, questions, truth)
            rows.append({'condition': condition, 'method': model_name, 'players': len(ids), **values,
                         'candidate_games_per_player': len(candidates)//len(ids), 'query_games_per_player': len(queries)//len(ids)})
            del model
        print(f'Color control {condition}: {len(ids)} matched players', flush=True)
    write_json(audit, directory / 'audit.json')
    write_csv(pd.DataFrame(coverage), directory / 'feature_coverage.csv')
    write_csv(pd.DataFrame(rows), ROOT / config['output_dir'] / 'color_control_results.csv')
    return rows, audit


def run(config):
    """Execute DEV2-only experiments and freeze the highest validation score."""
    started = time.perf_counter()
    directory = ROOT / config['output_dir']
    directory.mkdir(parents=True, exist_ok=True)
    # Saved CLOSED predictions are read once ONLY for identity exclusion metadata.
    audit = prepare_dev2(config)
    with dev2_only():
        train = read_csv(config['paths']['metric_train_csv'], ['player_id', 'game_id', 'color', 'sgf_content'])
        candidates = read_csv(config['paths']['val_candidates_csv'], ['player_id', 'game_id', 'color', 'sgf_content'])
        queries = read_csv(config['paths']['val_queries_csv'], ['question_id', 'game_id', 'color', 'sgf_content'])
        truth = read_csv(config['paths']['val_ground_truth_csv'], ['question_id', 'player_id'])
        pool = read_csv(directory / 'splits/val_pool.csv', ['player_id', 'game_id', 'color', 'sgf_content'])
        opening_rows, openings = opening_runs(directory, candidates, queries, truth)
        opening_best = max(opening_rows, key=lambda r: r['competition_score'])
        control_opening = max((r for r in opening_rows if not r['color_aware']), key=lambda r: r['competition_score'])
        opening_scores_best, players, questions, *_ = openings[opening_best['method']]
        stores, coverage = [], []
        for part, frame, column, source_key, manifest_key in [
            ('train', train, 'player_id', 'metric_train_csv', 'train_manifest'),
            ('val_candidate', candidates, 'player_id', 'val_candidates_csv', 'val_candidate_manifest'),
            ('val_query', queries, 'question_id', 'val_queries_csv', 'val_query_manifest')]:
            store, cov = cached_store(config, part, frame, column, source_key, manifest_key)
            stores.append(store)
            coverage.append(cov)
        write_csv(pd.DataFrame(coverage), directory / 'feature_coverage.csv')
        master, cs, qs = stores
        colors = dict(zip(train.game_id, train.color))
        order = np.random.default_rng(config['seed']).permutation(sorted(set(train.player_id))).tolist()
        results, exposure_rows, diagnostic_rows, cohort_by_name = {}, [], [], {}
        for count in config['training']['exposure_players']:
            ids = set(order[:count])
            if len(ids) != count:
                raise ValueError('Insufficient exposure cohort; cannot reduce players silently')
            view = ViewStore(master, ids, colors)
            if len(view) != count*config['split']['train_games']:
                raise ValueError('Feature failure changed the fixed TRAIN games/player exposure')
            name = f'Triplet-Hard-{count}'
            row, scores, diag, path = train_experiment(config, name, view, cs, qs, truth, players, questions)
            results[name] = (row, scores, path)
            cohort_by_name[name] = ids
            exposure_rows.append(row)
            diagnostic_rows.append(diag)
            write_csv(pd.DataFrame(exposure_rows), directory / 'exposure_results.csv')
            write_csv(pd.DataFrame(diagnostic_rows), directory / 'embedding_diagnostics.csv')
        hard_best = max(exposure_rows, key=lambda r: r['competition_score'])
        counts = Counter((r['group_id'], colors[r['game_id']]) for r in master.records)
        min_games = config['training']['cross_color_min_games']
        eligible = {p for p in cohort_by_name[hard_best['experiment']] if all(counts[p, c] >= min_games for c in ['B', 'W'])}
        if len(eligible) < 2:
            raise ValueError('Insufficient CrossColor eligible players; no same-color fallback')
        write_json({'parent_experiment': hard_best['experiment'], 'parent_players': hard_best['train_players'],
                    'min_games_per_color': min_games, 'actual_players': len(eligible), 'player_ids': sorted(eligible),
                    'excluded_player_ids': sorted(cohort_by_name[hard_best['experiment']]-eligible)}, directory/'cross_color_cohort.json')
        view = ViewStore(master, eligible, colors)
        if eligible == cohort_by_name[hard_best['experiment']]:
            matched_name = hard_best['experiment']
        else:
            matched_name = 'Triplet-Hard-Matched'
            row, scores, diag, path = train_experiment(config, matched_name, view, cs, qs, truth, players, questions)
            results[matched_name] = (row, scores, path)
            diagnostic_rows.append(diag)
        cross_name = 'Triplet-Hard-CrossColor'
        row, scores, diag, path = train_experiment(config, cross_name, view, cs, qs, truth, players, questions, True)
        results[cross_name] = (row, scores, path)
        diagnostic_rows.append(diag)
        write_csv(pd.DataFrame([results[matched_name][0], row]), directory/'cross_color_triplet_results.csv')
        write_csv(pd.DataFrame(diagnostic_rows), directory/'embedding_diagnostics.csv')
        triplet_best_name = max(results, key=lambda name: results[name][0]['competition_score'])
        triplet_best, triplet_scores, triplet_path = results[triplet_best_name]
        control_model_names = list(dict.fromkeys([matched_name, cross_name, triplet_best_name]))
        _, color_audit = matched_color_runs(config, pool, (control_opening['method'], *openings[control_opening['method']][3:]),
                                            {name: results[name][2] for name in control_model_names})
        fusion_rows = []
        for alpha in config['fusion']['alphas']:
            scores = fuse(opening_scores_best, triplet_scores, alpha, config['fusion']['normalization'])
            if alpha in [0, 1]:
                original = triplet_scores if alpha == 0 else opening_scores_best
                if not rankings(scores, players, questions).player_id.equals(rankings(original, players, questions).player_id):
                    raise ValueError('Fusion endpoint changed ranking')
            values, _ = metrics(scores, players, questions, truth)
            fusion_rows.append({'alpha': alpha, **values})
        write_csv(pd.DataFrame(fusion_rows), directory/'fusion_results.csv')
        best_fusion = max(fusion_rows, key=lambda r: r['competition_score'])
        actual = dict(zip(truth.question_id, truth.player_id))
        oc = np.array([players[i] == actual[q] for q, i in zip(questions, opening_scores_best.argmax(axis=1))])
        tc = np.array([players[i] == actual[q] for q, i in zip(questions, triplet_scores.argmax(axis=1))])
        overlap = [{'category': name, 'count': int(mask.sum()), 'proportion': float(mask.mean())} for name, mask in
                   [('opening_only_correct', oc & ~tc), ('triplet_only_correct', ~oc & tc),
                    ('both_correct', oc & tc), ('both_wrong', ~oc & ~tc)]]
        write_csv(pd.DataFrame(overlap), directory/'error_overlap.csv')
        choices = [{'kind': 'opening', 'method': opening_best['method'], 'competition_score': opening_best['competition_score']},
                   {'kind': 'triplet', 'method': triplet_best_name, 'competition_score': triplet_best['competition_score']},
                   {'kind': 'fusion', 'method': 'Opening+Triplet', **best_fusion}]
        choice = max(choices, key=lambda r: r['competition_score'])  # ties prefer pure Opening, then Triplet
        selected = copy.deepcopy(config)
        selected['selection'] = {'frozen': True, 'criterion': 'DEV2 validation competition score only',
            **choice, 'opening': {k: opening_best[k] for k in ['method', 'mode', 'window', 'color_aware']},
            'triplet': {'experiment': triplet_best_name, 'best_epoch': triplet_best['best_epoch'],
                'checkpoint': str(triplet_path.relative_to(ROOT)).replace('\\', '/'),
                'checkpoint_sha256': file_digest(triplet_path)},
            'fusion_normalization': config['fusion']['normalization'], 'source_sha256': audit['source_sha256'],
            'split_sha256': audit['file_sha256'], 'no_final_test_created_or_evaluated': True}
        (ROOT/'configs/phase28_selected.yaml').write_text(yaml.safe_dump(selected, allow_unicode=True, sort_keys=False), encoding='utf-8')
        summary = {'status': 'COMPLETE DEV2 ONLY', 'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.perf_counter()-started, 'split_audit': audit, 'feature_coverage': coverage,
            'opening_best': opening_best, 'triplet_best': triplet_best, 'best_fusion': best_fusion,
            'selected': choice, 'color_control_players': color_audit['players'], 'error_overlap': overlap,
            'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'deterministic': torch.are_deterministic_algorithms_enabled()},
            'closed_tests_accessed_for_inference_or_truth': False, 'final_test3_created': False}
        write_json(summary, directory/'summary.json')
        print('PHASE 2.8 — DEV2 RESULT\n'+json.dumps(choice, ensure_ascii=False, indent=2), flush=True)
        return summary


def main():
    """Run the configured, fixed DEV2 protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/phase28.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    directory = ROOT/config['output_dir']
    directory.mkdir(parents=True, exist_ok=True)
    class Tee:
        """Keep a complete UTF-8 real-run log and live progress output."""
        def __init__(self, console, stream):
            self.console, self.stream = console, stream
        def write(self, value):
            self.console.write(value)
            self.stream.write(value)
            self.stream.flush()
        def flush(self):
            self.console.flush()
            self.stream.flush()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    with (directory/'run.log').open('a', encoding='utf-8') as stream:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try:
            run(config)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == '__main__':
    main()
