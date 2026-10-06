"""Fresh DEV2 split with explicit metadata-only exclusion of previously evaluated players."""
import itertools

import numpy as np
import pandas as pd

from .experiment_state import read_json
from .metric_utils import file_digest, write_json
from .utils import ROOT, read_csv, write_csv


def exclusion_ids():
    """Recover closed player pools from saved predictions/IDs, never CLOSED truths."""
    receipt = read_json('outputs/phase26/final_test2/final_test2_receipt.json')
    if receipt['evaluation_count'] != 1 or receipt['status'] != 'CLOSED TEST':
        raise ValueError('FINAL TEST 2 must already be permanently closed')
    paths = ['outputs/phase26/final_test2/random_predictions.csv',
             'outputs/phase26/final_test2/opening10_predictions.csv',
             'outputs/phase26/final_test2/triplet_predictions.csv']
    final_ids = set()
    provenance = {}
    for path in paths:
        final_ids.update(pd.read_csv(ROOT/path, dtype=str, usecols=['player_id']).player_id)
        provenance[path] = file_digest(path)
    if len(final_ids) != receipt['final_players']:
        raise ValueError('Saved predictions do not expose the complete 50-player pool; do not reopen closed inputs')
    old = set(read_json('outputs/phase26/splits/split_audit.json')['closed_round1_excluded_players'])
    val = set(pd.read_csv(ROOT/'outputs/phase26/splits/dev_val_candidates.csv', dtype=str, usecols=['player_id']).player_id)
    if len(old) != 10 or len(val) != 50 or old & final_ids or val & final_ids or old & val:
        raise ValueError('Previous cohort metadata inconsistent')
    return old | val | final_ids, {'round1_test': sorted(old), 'phase26_val': sorted(val),
                                 'final_test2': sorted(final_ids), 'prediction_sha256': provenance}


def make_dev2(source, excluded, settings, seed=42):
    """Choose VAL first from eligible identities, then disjoint TRAIN; never lower counts."""
    if source.game_id.duplicated().any() or source.sgf_content.duplicated().any():
        raise ValueError('Duplicate source game/SGF requires explicit handling before DEV2')
    source = source[~source.player_id.isin(excluded)].copy()
    counts = source.groupby('player_id').size()
    rng = np.random.default_rng(seed)
    eligible_val = np.array(sorted(counts[counts >= settings['candidate_games'] + settings['query_games']].index))
    if len(eligible_val) < settings['val_players']:
        raise ValueError('Insufficient eligible DEV2 VAL players')
    val_ids = set(rng.permutation(eligible_val)[:settings['val_players']])
    eligible_train = np.array(sorted(p for p in counts[counts >= settings['train_games']].index if p not in val_ids))
    if len(eligible_train) < settings['train_players']:
        raise ValueError('Insufficient eligible DEV2 TRAIN players')
    train_ids = set(rng.permutation(eligible_train)[:settings['train_players']])
    train, candidates, queries, truths = [], [], [], []
    for player in sorted(train_ids | val_ids):
        group = source[source.player_id == player].sort_values('game_id')
        selected = group.iloc[rng.permutation(len(group))]
        if player in train_ids:
            train.append(selected.iloc[:settings['train_games']])
        else:
            question = f'dev2_q_{len(truths):04d}'
            n, m = settings['candidate_games'], settings['query_games']
            candidates.append(selected.iloc[:n])
            queries.append(selected.iloc[n:n+m][['game_id', 'color', 'sgf_content']].assign(question_id=question))
            truths.append({'question_id': question, 'player_id': player})
    frames = [pd.concat(items, ignore_index=True) for items in [train, candidates, queries]]
    frames.append(pd.DataFrame(truths))
    audit = {'train_players': len(train_ids), 'val_players': len(val_ids), 'player_overlap': len(train_ids & val_ids),
             'excluded_overlap': len((train_ids | val_ids) & excluded), 'game_id_overlap': {}, 'sgf_content_overlap': {}}
    for a, b in itertools.combinations(range(3), 2):
        for field in ['game_id', 'sgf_content']:
            audit[field + '_overlap'][f'{a}/{b}'] = len(set(frames[a][field]) & set(frames[b][field]))
    if audit['player_overlap'] or audit['excluded_overlap'] or any(audit['game_id_overlap'].values()) or any(audit['sgf_content_overlap'].values()):
        raise ValueError('DEV2 leakage')
    return frames, source[source.player_id.isin(val_ids)], audit


def prepare_dev2(config):
    """Create once, or verify the existing split without re-sampling identities."""
    excluded, provenance = exclusion_ids()
    directory = ROOT / config['output_dir'] / 'splits'
    audit_path = directory / 'audit.json'
    names = ['train.csv', 'val_candidates.csv', 'val_queries.csv', 'val_ground_truth.csv', 'val_pool.csv']
    source_hash = file_digest(config['paths']['training_csv'])
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit['source_sha256'] != source_hash or audit['seed'] != config['seed'] or audit['settings'] != config['split'] or audit['excluded_provenance'] != provenance:
            raise ValueError('Existing DEV2 provenance changed; refusing a new split')
        for name in names:
            if file_digest(directory / name) != audit['file_sha256'][name]:
                raise ValueError('DEV2 CSV modified')
        return audit
    source = read_csv(config['paths']['training_csv'], ['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
    if not set(source.color) <= {'B', 'W'} or not excluded <= set(source.player_id):
        raise ValueError('Unknown color or excluded player identity')
    frames, pool, audit = make_dev2(source, excluded, config['split'], config['seed'])
    for name, frame in zip(names, frames + [pool]):
        write_csv(frame, directory / name)
    audit.update(source_sha256=source_hash, seed=config['seed'], settings=config['split'], excluded_provenance=provenance,
                 file_sha256={name: file_digest(directory / name) for name in names})
    write_json(audit, audit_path)
    return audit


def color_cohort(pool, candidate_games=30, query_games=10, seed=42):
    """Share one eligible player pool and equal game budgets across all five color controls."""
    counts = pool.groupby(['player_id', 'color']).size().unstack(fill_value=0)
    needed = candidate_games + query_games
    ids = sorted(counts[(counts.get('B', 0) >= needed) & (counts.get('W', 0) >= needed)].index)
    if len(ids) < 2:
        raise ValueError('Fewer than two eligible players for matched color controls')
    rng, sampled = np.random.default_rng(seed), {}
    for player in ids:
        for color in ['B', 'W']:
            group = pool[(pool.player_id == player) & (pool.color == color)].sort_values('game_id')
            sampled[player, color] = group.iloc[rng.permutation(len(group))[:needed]]
    output = []
    for name, cc, qc in [('B_to_B', 'B', 'B'), ('W_to_W', 'W', 'W'), ('B_to_W', 'B', 'W'),
                         ('W_to_B', 'W', 'B'), ('Mixed_to_Mixed', None, None)]:
        cs, qs, ts = [], [], []
        for index, player in enumerate(ids):
            if cc:
                c = sampled[player, cc].iloc[:candidate_games]
                q = sampled[player, qc].iloc[candidate_games:candidate_games+query_games]
            else:
                c = pd.concat([sampled[player, 'B'].iloc[:candidate_games//2], sampled[player, 'W'].iloc[:candidate_games-candidate_games//2]])
                q = pd.concat([sampled[player, 'B'].iloc[candidate_games:candidate_games+query_games//2],
                               sampled[player, 'W'].iloc[candidate_games:candidate_games+query_games-query_games//2]])
            question = f'color_q_{index:04d}'
            cs.append(c)
            qs.append(q[['game_id', 'color', 'sgf_content']].assign(question_id=question))
            ts.append({'question_id': question, 'player_id': player})
        c, q, t = pd.concat(cs, ignore_index=True), pd.concat(qs, ignore_index=True), pd.DataFrame(ts)
        if set(c.game_id) & set(q.game_id) or set(c.sgf_content) & set(q.sgf_content):
            raise ValueError('Color control overlap')
        output.append((name, c, q, t))
    return output, ids
