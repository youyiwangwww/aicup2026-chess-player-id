"""Build reproducible disjoint candidate and query sets."""
import numpy as np
import pandas as pd
from .utils import argument_parser, load_config, read_csv, write_csv


def build_validation(frame, num_players, candidate_games, query_games, seed):
    """Sample unique games; globally exclude candidate/query game overlap."""
    if min(num_players, candidate_games, query_games) < 1:
        raise ValueError('Player and game counts must be positive')
    frame = frame.copy()
    # Conservative identity: duplicate IDs OR identical SGFs denote the same game.
    frame = frame.drop_duplicates(['player_id', 'game_id'])
    frame = frame.drop_duplicates(['player_id', 'sgf_content'])
    frame = frame.sort_values(['player_id', 'game_id', 'sgf_content']).reset_index(drop=True)
    required = candidate_games + query_games
    counts = frame.groupby('player_id').size()
    eligible = sorted(counts[counts >= required].index)
    if len(eligible) < num_players:
        raise ValueError(f'Need {num_players} eligible players, found {len(eligible)} '
                         f'with at least {required} distinct games')
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, num_players, replace=False)
    candidates, queries, truths = [], [], []
    candidate_ids, candidate_sgfs, query_ids, query_sgfs = set(), set(), set(), set()
    for i, player in enumerate(selected, 1):
        pool = frame[frame.player_id == player]
        pool = pool[~pool.game_id.isin(query_ids) & ~pool.sgf_content.isin(query_sgfs)]
        if len(pool) < required:
            raise ValueError('Shared games prevent a disjoint split; try another seed or fewer players')
        chosen = pool.iloc[rng.permutation(len(pool))]
        candidate = chosen.iloc[:candidate_games]
        candidate_ids.update(candidate.game_id)
        candidate_sgfs.update(candidate.sgf_content)
        remaining = chosen.iloc[candidate_games:]
        remaining = remaining[~remaining.game_id.isin(candidate_ids)
                              & ~remaining.sgf_content.isin(candidate_sgfs)]
        if len(remaining) < query_games:
            raise ValueError('Shared games prevent a disjoint split; try another seed or fewer players')
        query = remaining.iloc[:query_games]
        query_ids.update(query.game_id)
        query_sgfs.update(query.sgf_content)
        question = f'q_{i:04d}'
        candidates.append(candidate[['player_id', 'game_id', 'color', 'sgf_content']])
        # Only fields needed for inference; no player IDs or ranks leak into queries.
        queries.append(query[['game_id', 'color', 'sgf_content']].assign(question_id=question))
        truths.append({'question_id': question, 'player_id': player})
    return pd.concat(candidates, ignore_index=True), pd.concat(queries, ignore_index=True), pd.DataFrame(truths)


def main():
    """Read parameters and save the three validation CSVs."""
    parser = argument_parser(__doc__)
    parser.add_argument('--input')
    parser.add_argument('--num-players', type=int)
    parser.add_argument('--candidate-games', type=int)
    parser.add_argument('--query-games', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config['validation'].copy()
    for key in settings:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    frame = read_csv(args.input or config['paths']['training_csv'],
                     ['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
    result = build_validation(frame, **settings, seed=args.seed if args.seed is not None else config['seed'])
    for name, data in zip(['candidates_csv', 'queries_csv', 'ground_truth_csv'], result):
        write_csv(data, config['paths'][name])
        print(f'{name}: {len(data)} rows')


if __name__ == '__main__':
    main()
