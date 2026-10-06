"""Create reproducible cross-color DEV validation and evaluate fixed methods."""
import copy

import numpy as np
import pandas as pd

from .analyze_opening_signal import opening_metrics
from .embed_players import load_encoder
from .evaluate import evaluate
from .metric_utils import choose_device, seed_everything, write_json
from .phase26_common import MemoryStore, dev_only, dev_retrieve
from .player_features import preprocess_partition
from .utils import read_csv, write_csv


def cross_color_partitions(pool, candidate_games=20, query_games=10, seed=42):
    """Use identical eligible DEV players in both directions without lowering game counts."""
    pool = pool.copy()
    pool['color'] = pool.color.str.upper()
    counts = pool.groupby(['player_id', 'color']).size().unstack(fill_value=0)
    minimum = max(candidate_games, query_games)
    eligible = sorted(counts[(counts.get('B', 0) >= minimum) & (counts.get('W', 0) >= minimum)].index)
    if len(eligible) < 2:
        return [], {'status': 'insufficient_cross_color_players', 'eligible_players': len(eligible)}
    rng = np.random.default_rng(seed)
    sampled = {}
    for player in eligible:
        group = pool[pool.player_id == player]
        for color in ['B', 'W']:
            games = group[group.color == color].sort_values('game_id')
            sampled[player, color] = games.iloc[rng.permutation(len(games))[:minimum]]
    partitions = []
    for candidate_color, query_color in [('B', 'W'), ('W', 'B')]:
        candidates, queries, truths = [], [], []
        for index, player in enumerate(eligible):
            question = f'cross_q_{index:04d}'
            candidates.append(sampled[player, candidate_color].iloc[:candidate_games])
            queries.append(sampled[player, query_color].iloc[:query_games]
                           [['game_id', 'color', 'sgf_content']].assign(question_id=question))
            truths.append({'question_id': question, 'player_id': player})
        candidates = pd.concat(candidates, ignore_index=True)
        queries = pd.concat(queries, ignore_index=True)
        if set(candidates.game_id) & set(queries.game_id) or set(candidates.sgf_content) & set(queries.sgf_content):
            raise ValueError('Cross-color candidate/query overlap')
        partitions.append((f'{candidate_color}_to_{query_color}', candidates, queries, pd.DataFrame(truths)))
    return partitions, {'status': 'available', 'eligible_players': len(eligible),
                        'dev_val_players': pool.player_id.nunique(), 'candidate_games_per_player': candidate_games,
                        'query_games_per_player': query_games, 'game_id_overlap': 0, 'sgf_overlap': 0,
                        'note': 'DEV-only eligible subset; directions reuse games across experiments.'}


def analyze_cross_color(base, selected):
    """Assess color transfer after fixing DEV-selected settings; no model selection here."""
    with dev_only():
        pool = read_csv('outputs/phase26/splits/dev_val_pool.csv', ['player_id', 'game_id', 'color', 'sgf_content'])
        partitions, audit = cross_color_partitions(pool,
            base['diagnosis']['cross_color_candidate_games'], base['diagnosis']['cross_color_query_games'], base['seed'])
        write_json(audit, 'outputs/phase26/cross_color_audit.json')
        rows = []
        seed_everything(selected['seed'], True, selected['training']['num_threads'])
        device = choose_device(selected['training']['device'])
        model, _ = load_encoder(selected['paths']['best_checkpoint'], selected, device)
        for direction, candidates, queries, truth in partitions:
            metrics, errors = opening_metrics(candidates, queries, truth, 40)
            write_csv(errors, f'outputs/phase26/cross_color/{direction}/opening_errors.csv')
            rows.append({'direction': direction, 'method': 'Opening-40', 'players': len(truth), **metrics})
            config = copy.deepcopy(selected)
            directory = f'outputs/phase26/cross_color/{direction}'
            config['paths'].update(val_candidates_csv=f'{directory}/candidates.csv',
                val_queries_csv=f'{directory}/queries.csv', cache_dir=f'{directory}/features',
                val_candidate_manifest=f'{directory}/candidate.json', val_query_manifest=f'{directory}/query.json')
            for frame, group, key, manifest in [(candidates, 'player_id', 'val_candidates_csv', 'val_candidate_manifest'),
                                              (queries, 'question_id', 'val_queries_csv', 'val_query_manifest')]:
                write_csv(frame, config['paths'][key])
                preprocess_partition(frame, group, config['paths'][key], config['paths'][manifest], config)
            predictions, _, _ = dev_retrieve(model, MemoryStore(config, 'val_candidate'),
                                            MemoryStore(config, 'val_query'), device, config['inference']['batch_size'])
            metrics, _ = evaluate(predictions, truth)
            rows.append({'direction': direction, 'method': 'Triplet-selected', 'players': len(truth), **metrics})
        result = pd.DataFrame(rows, columns=['direction', 'method', 'players', 'top_1_accuracy',
                                           'top_3_accuracy', 'top_5_accuracy', 'competition_score'])
        write_csv(result, 'outputs/phase26/cross_color_results.csv')
        print(result.to_string(index=False), flush=True)
        return result
