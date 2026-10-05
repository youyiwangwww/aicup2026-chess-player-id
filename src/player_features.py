"""Sample pre-move Go histories and preprocess bounded compressed NPZ shards."""
from collections import deque
import hashlib
import json

import numpy as np
import pandas as pd
from sgfmill import boards, sgf

from .build_metric_split import read_metric_partitions
from .metric_utils import file_digest, metric_parser, write_json
from .utils import load_config, resolve_path, write_csv

FEATURE_VERSION = 1


def board_planes(board):
    """Encode stones with sgfmill coordinates: row 0 bottom, column 0 left."""
    black = np.zeros((19, 19), dtype=np.uint8)
    white = np.zeros_like(black)
    for color, (row, column) in board.list_occupied_points():
        (black if color == 'b' else white)[row, column] = 1
    return black, white


def extract_player_features(sgf_content, target_color, history_length=8,
                            max_positions_per_game=4, seed=42, board_size=19):
    """Return sampled (positions, 2*history+1, 19, 19) before target turns.

    Histories are oldest to newest and left-padded with empty boards. Pass is
    a target turn and advances history; setup nodes do not advance move count.
    At most K feature tensors are built, regardless of game length.
    """
    if history_length < 1 or max_positions_per_game < 1 or board_size != 19 or seed < 0:
        raise ValueError('Require history>=1, max_positions>=1, board_size=19 and seed>=0')
    target = str(target_color).strip().lower()
    if target not in ('b', 'w'):
        raise ValueError(f'Invalid target color: {target_color!r}')
    try:
        game = sgf.Sgf_game.from_string(sgf_content)
        if game.get_size() != board_size:
            raise ValueError(f'Expected 19x19 board, got {game.get_size()}')
        nodes = game.get_main_sequence()
        move_nodes = [(i, node.get_move()[0]) for i, node in enumerate(nodes)]
        target_indices = [i for i, color in move_nodes if color == target]
        if not target_indices:
            raise ValueError('No target-player turns')
        # Only content/color/seed determine sampling; no player or rank labels.
        material = f'{seed}\0{target}\0{sgf_content}'.encode('utf-8')
        local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], 'little')
        rng = np.random.default_rng(local_seed)
        selected = set(rng.choice(target_indices, min(max_positions_per_game, len(target_indices)),
                                  replace=False).tolist())
        board = boards.Board(board_size)
        history = deque(maxlen=history_length)
        features = []
        for index, node in enumerate(nodes):
            black_setup, white_setup, empty_setup = node.get_setup_stones()
            if black_setup or white_setup or empty_setup:
                if not board.apply_setup(black_setup, white_setup, empty_setup):
                    raise ValueError('Invalid setup stones')
            color, move = node.get_move()
            if color is None:
                continue
            history.append(board_planes(board))
            if index in selected:
                own_index = 0 if target == 'b' else 1
                empty = np.zeros((board_size, board_size), dtype=np.uint8)
                padding = [empty] * (history_length - len(history))
                own = padding + [state[own_index] for state in history]
                opponent = padding + [state[1 - own_index] for state in history]
                color_plane = np.full_like(empty, int(target == 'b'))
                features.append(np.stack(own + opponent + [color_plane]))
            if move is not None:
                board.play(move[0], move[1], color)
        return np.stack(features).astype(np.uint8, copy=False)
    except Exception as exc:
        raise ValueError(f'SGF/board replay failed: {exc}') from exc


def preprocess_partition(frame, group_column, csv_path, manifest_path, config):
    """Write game-batched shards and a source/settings-bound cache manifest."""
    settings = config['features']
    shard_games = config['cache']['games_per_shard']
    if shard_games < 1:
        raise ValueError('games_per_shard must be positive')
    source_hash = file_digest(csv_path)
    signature = {'source_sha256': source_hash, 'features': settings,
                 'seed': config['seed'], 'feature_version': FEATURE_VERSION}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    directory = resolve_path(config['paths']['cache_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    prefix = resolve_path(manifest_path).stem
    records, errors, pending, pending_rows = [], [], [], []
    shard_index = 0

    def flush():
        """Keep peak preprocessing memory bounded by games_per_shard."""
        nonlocal shard_index
        if not pending:
            return
        name = f'{prefix}_{key}_{shard_index:05d}.npz'
        offsets = np.cumsum([0] + [len(array) for array in pending], dtype=np.int64)
        np.savez_compressed(directory / name, features=np.concatenate(pending), offsets=offsets)
        for slot, row in enumerate(pending_rows):
            records.append({**row, 'shard': name, 'slot': slot,
                            'num_positions': int(offsets[slot + 1] - offsets[slot])})
        pending.clear()
        pending_rows.clear()
        shard_index += 1

    for row in frame.itertuples(index=False):
        identity = str(getattr(row, group_column))
        try:
            array = extract_player_features(row.sgf_content, row.color, **settings, seed=config['seed'])
        except ValueError as exc:
            errors.append({'partition': prefix, 'group_id': identity,
                           'game_id': row.game_id, 'error': str(exc)})
            continue
        pending.append(array)
        pending_rows.append({'game_id': row.game_id, 'group_id': identity})
        if len(pending) >= shard_games:
            flush()
    flush()
    manifest = {**signature, 'group_column': group_column, 'cache_key': key,
                'input_games': len(frame), 'valid_games': len(records),
                'skipped_games': len(errors), 'shards': shard_index,
                'cache_directory': str(directory), 'records': records}
    write_json(manifest, manifest_path)
    print(f'{prefix}: {len(records)}/{len(frame)} valid games; '
          f'{sum(r["num_positions"] for r in records)} positions; {shard_index} shards', flush=True)
    return manifest, errors


def main():
    """Audit inputs, preprocess all three partitions, and record per-game errors."""
    args = metric_parser(__doc__).parse_args()
    config = load_config(args.config)
    training, candidates, queries, _ = read_metric_partitions(config)
    jobs = [(training, 'player_id', 'metric_train_csv', 'train_manifest'),
            (candidates, 'player_id', 'candidates_csv', 'candidate_manifest'),
            (queries, 'question_id', 'queries_csv', 'query_manifest')]
    errors = []
    for frame, column, source_key, manifest_key in jobs:
        _, skipped = preprocess_partition(frame, column, config['paths'][source_key],
                                          config['paths'][manifest_key], config)
        errors.extend(skipped)
    write_csv(pd.DataFrame(errors, columns=['partition', 'group_id', 'game_id', 'error']),
              config['paths']['feature_errors_csv'])


if __name__ == '__main__':
    main()
