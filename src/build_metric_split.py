"""Split Go players into disjoint metric-training and held-out retrieval groups."""
import itertools

import numpy as np
import pandas as pd

from .metric_utils import file_digest, metric_parser, write_json
from .utils import load_config, read_csv, write_csv

THREE_WAY_KEYS = ['metric_train_csv', 'val_candidates_csv', 'val_queries_csv',
                  'val_ground_truth_csv', 'test_candidates_csv', 'test_queries_csv',
                  'test_ground_truth_csv']


def is_three_way(config):
    """Distinguish Phase 2.5 configs from the preserved two-way experiments."""
    return 'val_players' in config['split']


def audit_three_way_split(training, val_candidates, val_queries, val_truth,
                          test_candidates, test_queries, test_truth, seed):
    """Audit three identity pools and all ten pairs of game partitions."""
    audit_split(training, val_candidates, val_queries, val_truth, seed)
    audit_split(training, test_candidates, test_queries, test_truth, seed)
    players = {'train': set(training.player_id), 'val': set(val_truth.player_id),
               'test': set(test_truth.player_id)}
    player_overlaps = {f'{a}_{b}': len(players[a] & players[b])
                       for a, b in itertools.combinations(players, 2)}
    frames = {'train': training, 'val_candidate': val_candidates, 'val_query': val_queries,
              'test_candidate': test_candidates, 'test_query': test_queries}
    game_overlaps, sgf_overlaps = {}, {}
    for (a, left), (b, right) in itertools.combinations(frames.items(), 2):
        game_overlaps[f'{a}_{b}'] = len(set(left.game_id) & set(right.game_id))
        sgf_overlaps[f'{a}_{b}'] = len(set(left.sgf_content) & set(right.sgf_content))
    if any(player_overlaps.values()) or any(game_overlaps.values()) or any(sgf_overlaps.values()):
        raise ValueError(f'Data leakage in three-way split: players={player_overlaps}, '
                         f'game_ids={game_overlaps}, SGFs={sgf_overlaps}')
    return {'schema_version': 25, 'seed': int(seed),
            **{f'{name}_player_count': len(ids) for name, ids in players.items()},
            **{f'{name}_game_count': len(frame) for name, frame in frames.items()},
            'player_overlap_counts': player_overlaps, 'game_id_overlap_counts': game_overlaps,
            'sgf_content_overlap_counts': sgf_overlaps,
            'sgf_identity': 'exact sgf_content string; not canonicalized'}


def build_three_way_split(frame, train_players, val_players, test_players,
                          train_games_per_player, val_candidate_games, val_query_games,
                          test_candidate_games, test_query_games, seed=42):
    """Select fixed-sized identity pools, allocating the strictest game requirement first."""
    values = [train_players, val_players, test_players, train_games_per_player,
              val_candidate_games, val_query_games, test_candidate_games, test_query_games]
    if min(values) < 1 or train_players < 2 or train_games_per_player < 2 or seed < 0:
        raise ValueError('Positive split counts and seed>=0 required; TRAIN needs >=2 players/games')
    frame = frame.drop_duplicates(['player_id', 'game_id'])
    frame = frame.drop_duplicates(['player_id', 'sgf_content'])
    frame = frame.sort_values(['player_id', 'game_id', 'sgf_content']).reset_index(drop=True)
    counts = frame.groupby('player_id').size()
    specifications = [('train', train_players, train_games_per_player),
                      ('val', val_players, val_candidate_games + val_query_games),
                      ('test', test_players, test_candidate_games + test_query_games)]
    rng = np.random.default_rng(seed)
    selected, used = {}, set()
    for name, number, minimum in sorted(specifications, key=lambda item: -item[2]):
        eligible = sorted(p for p in counts[counts >= minimum].index if p not in used)
        if len(eligible) < number:
            raise ValueError(f'Need {number} {name} players with >= {minimum} distinct games '
                             f'and disjoint identities; found {len(eligible)} remaining eligible players')
        selected[name] = rng.choice(eligible, number, replace=False).tolist()
        used.update(selected[name])
    groups = {p: group for p, group in frame.groupby('player_id', sort=True)}
    cols = ['player_id', 'game_id', 'color', 'sgf_content']
    training = pd.concat([groups[p].iloc[rng.permutation(len(groups[p]))[:train_games_per_player]]
                          for p in selected['train']], ignore_index=True)[cols]
    partitions = [training]
    for name, candidate_count, query_count in [('val', val_candidate_games, val_query_games),
                                              ('test', test_candidate_games, test_query_games)]:
        candidates, queries, truths = [], [], []
        for index, player in enumerate(selected[name], 1):
            pool = groups[player]
            games = pool.iloc[rng.permutation(len(pool))[:candidate_count + query_count]]
            question = f'{name}_q_{index:04d}'
            candidates.append(games.iloc[:candidate_count][cols])
            queries.append(games.iloc[candidate_count:][['game_id', 'color', 'sgf_content']]
                           .assign(question_id=question))
            truths.append({'question_id': question, 'player_id': player})
        partitions.extend([pd.concat(candidates, ignore_index=True),
                           pd.concat(queries, ignore_index=True), pd.DataFrame(truths)])
    audit = audit_three_way_split(*partitions, seed)
    return (*partitions, audit)


def verify_partition_hashes(config, keys):
    """Check only explicitly requested CSVs against the split-stage audit."""
    import json
    from .utils import resolve_path

    with resolve_path(config['paths']['split_audit_json']).open(encoding='utf-8') as stream:
        audit = json.load(stream)
    if audit.get('schema_version') != 25 or audit['seed'] != config['seed']:
        raise ValueError('Split audit/config mismatch; rebuild the three-way split')
    for key in keys:
        if audit['partition_sha256'][key] != file_digest(config['paths'][key]):
            raise ValueError(f'{key} changed since split audit; rebuild the experiment')
    return audit


def read_training_validation(config):
    """Read TRAIN and VALIDATION only; never open any TEST CSV or cache."""
    if not is_three_way(config):
        return read_metric_partitions(config)
    verify_partition_hashes(config, THREE_WAY_KEYS[:4])
    paths = config['paths']
    cols = ['player_id', 'game_id', 'color', 'sgf_content']
    training = read_csv(paths['metric_train_csv'], cols)
    candidates = read_csv(paths['val_candidates_csv'], cols)
    queries = read_csv(paths['val_queries_csv'], ['question_id', 'game_id', 'color', 'sgf_content'])
    truth = read_csv(paths['val_ground_truth_csv'], ['question_id', 'player_id'])
    audit_split(training, candidates, queries, truth, config['seed'])
    return training, candidates, queries, truth


def read_three_way_partitions(config):
    """Read all seven partitions only in the split/preprocessing stage."""
    paths = config['paths']
    verify_partition_hashes(config, THREE_WAY_KEYS)
    training, val_candidates, val_queries, val_truth = read_training_validation(config)
    cols = ['player_id', 'game_id', 'color', 'sgf_content']
    test_candidates = read_csv(paths['test_candidates_csv'], cols)
    test_queries = read_csv(paths['test_queries_csv'], ['question_id', 'game_id', 'color', 'sgf_content'])
    test_truth = read_csv(paths['test_ground_truth_csv'], ['question_id', 'player_id'])
    partitions = (training, val_candidates, val_queries, val_truth, test_candidates, test_queries, test_truth)
    audit_three_way_split(*partitions, config['seed'])
    return partitions


def audit_split(training, candidates, queries, truth, seed):
    """Raise on player/game/SGF leakage, even if CSVs were edited after splitting."""
    if any(frame.empty for frame in [training, candidates, queries, truth]):
        raise ValueError('All metric split partitions must be nonempty')
    if 'player_id' in queries.columns or 'rank' in queries.columns:
        raise ValueError('Queries must not contain player_id or rank')
    if truth.question_id.duplicated().any() or truth.player_id.duplicated().any():
        raise ValueError('Metric validation needs exactly one question per held-out player')
    if set(queries.question_id) != set(truth.question_id):
        raise ValueError('Query question IDs and ground truth do not match')
    if set(candidates.player_id) != set(truth.player_id):
        raise ValueError('Candidate player IDs and ground truth do not match')
    overlaps = {}
    train_players = set(training.player_id)
    overlaps['train_candidate_player_id'] = len(train_players & set(candidates.player_id))
    overlaps['train_query_player_id'] = len(train_players & set(truth.player_id))
    frames = {'train': training, 'candidate': candidates, 'query': queries}
    for (left, a), (right, b) in itertools.combinations(frames.items(), 2):
        for column in ('game_id', 'sgf_content'):
            overlaps[f'{left}_{right}_{column}'] = len(set(a[column]) & set(b[column]))
    if any(overlaps.values()):
        raise ValueError(f'Data leakage detected; all overlaps must be zero: {overlaps}')
    for name, frame in frames.items():
        if frame.game_id.duplicated().any() or frame.sgf_content.duplicated().any():
            raise ValueError(f'Duplicate game_id or sgf_content inside {name} split')
    return {'seed': int(seed), 'training_player_count': len(train_players),
            'evaluation_player_count': truth.player_id.nunique(),
            'training_game_count': len(training), 'candidate_game_count': len(candidates),
            'query_game_count': len(queries), 'overlap_counts': overlaps,
            'sgf_identity': 'exact sgf_content string; not canonicalized'}


def build_metric_split(frame, train_players, eval_players, train_games_per_player,
                       candidate_games, query_games, seed=42):
    """Sample fixed counts without reducing requests or hiding cross-partition leakage."""
    if min(train_players, eval_players, candidate_games, query_games) < 1:
        raise ValueError('Player and game counts must be positive')
    if train_players < 2 or train_games_per_player < 2:
        raise ValueError('Triplet training requires >=2 players and >=2 games/player')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    frame = frame.drop_duplicates(['player_id', 'game_id'])
    frame = frame.drop_duplicates(['player_id', 'sgf_content'])
    frame = frame.sort_values(['player_id', 'game_id', 'sgf_content']).reset_index(drop=True)
    counts = frame.groupby('player_id').size()
    eligible_eval = sorted(counts[counts >= candidate_games + query_games].index)
    eligible_train = set(counts[counts >= train_games_per_player].index)
    if len(eligible_eval) < eval_players:
        raise ValueError(f'Need {eval_players} evaluation players with >= '
                         f'{candidate_games + query_games} distinct games; found {len(eligible_eval)}')
    # If evaluation players use up training-eligible IDs, reserve exactly enough.
    eval_only = [p for p in eligible_eval if p not in eligible_train]
    shared_needed = max(0, eval_players - len(eval_only))
    if len(eligible_train) - shared_needed < train_players:
        raise ValueError(f'Cannot select {train_players} training and {eval_players} evaluation '
                         f'players with disjoint identities; training-eligible={len(eligible_train)}, '
                         f'evaluation-eligible={len(eligible_eval)}')
    rng = np.random.default_rng(seed)
    # Prefer eval-only players only when needed to keep the requested training count.
    max_shared = len(eligible_train) - train_players
    eval_order = list(rng.permutation(eligible_eval))
    selected_eval = []
    used_shared = 0
    for player in eval_order:
        if player in eligible_train and used_shared >= max_shared:
            continue
        selected_eval.append(player)
        used_shared += int(player in eligible_train)
        if len(selected_eval) == eval_players:
            break
    selected_train = rng.choice(sorted(eligible_train - set(selected_eval)), train_players, replace=False)
    groups = {p: group for p, group in frame.groupby('player_id', sort=True)}
    training, candidates, queries, truths = [], [], [], []
    for player in selected_train:
        pool = groups[player]
        training.append(pool.iloc[rng.permutation(len(pool))[:train_games_per_player]])
    for index, player in enumerate(selected_eval, 1):
        pool = groups[player]
        selected = pool.iloc[rng.permutation(len(pool))[:candidate_games + query_games]]
        question = f'metric_q_{index:04d}'
        candidates.append(selected.iloc[:candidate_games])
        queries.append(selected.iloc[candidate_games:][['game_id', 'color', 'sgf_content']]
                       .assign(question_id=question))
        truths.append({'question_id': question, 'player_id': player})
    cols = ['player_id', 'game_id', 'color', 'sgf_content']
    training = pd.concat(training, ignore_index=True)[cols]
    candidates = pd.concat(candidates, ignore_index=True)[cols]
    queries = pd.concat(queries, ignore_index=True)
    truth = pd.DataFrame(truths)
    audit = audit_split(training, candidates, queries, truth, seed)
    return training, candidates, queries, truth, audit


def read_metric_partitions(config):
    """Read and re-audit the exact CSVs used by training and comparison."""
    paths = config['paths']
    cols = ['player_id', 'game_id', 'color', 'sgf_content']
    training = read_csv(paths['metric_train_csv'], cols)
    candidates = read_csv(paths['candidates_csv'], cols)
    queries = read_csv(paths['queries_csv'], ['question_id', 'game_id', 'color', 'sgf_content'])
    truth = read_csv(paths['ground_truth_csv'], ['question_id', 'player_id'])
    audit_split(training, candidates, queries, truth, config['seed'])
    return training, candidates, queries, truth


def main():
    """Write the audited split and input provenance for one rank-group CSV."""
    parser = metric_parser(__doc__)
    parser.add_argument('--input')
    args = parser.parse_args()
    config = load_config(args.config)
    source = args.input or config['paths']['training_csv']
    frame = read_csv(source, ['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
    builder = build_three_way_split if is_three_way(config) else build_metric_split
    result = builder(frame, **config['split'], seed=config['seed'])
    keys = THREE_WAY_KEYS if is_three_way(config) else [
        'metric_train_csv', 'candidates_csv', 'queries_csv', 'ground_truth_csv']
    audit = result[-1]
    audit['source_csv_sha256'] = file_digest(source)
    for key, partition in zip(keys, result[:-1]):
        write_csv(partition, config['paths'][key])
        print(f'{key}: {len(partition)} rows', flush=True)
    audit['partition_sha256'] = {key: file_digest(config['paths'][key]) for key in keys}
    write_json(audit, config['paths']['split_audit_json'])
    print(f'Player/game/SGF overlap audit passed: {audit.get("overlap_counts", audit.get("player_overlap_counts"))}',
          flush=True)


if __name__ == '__main__':
    main()
