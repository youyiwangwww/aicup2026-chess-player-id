"""Read-only safety checks for the frozen FINAL TEST 2 evaluation."""
import itertools

import pandas as pd
import torch

from .experiment_state import experiment_signature, read_json
from .final_test2_lock import require_unconsumed
from .metric_utils import file_digest
from .phase26_prepare import preserve_round1
from .utils import ROOT, load_config, read_csv, resolve_path


def require(condition, message):
    """Stop before evaluation when any safety invariant fails."""
    if not condition:
        raise ValueError(message)


def labeled_queries(queries, truth):
    """Validate the question mapping and restore identities solely for split auditing."""
    require('player_id' not in queries, 'Query CSV must not contain player_id')
    require(not truth.question_id.duplicated().any(), 'Duplicate ground-truth question')
    require(set(queries.question_id) == set(truth.question_id), 'Query/truth question mismatch')
    return queries.merge(truth, on='question_id', how='left', validate='many_to_one')


def check_final_counts(candidates, queries, truth, players=50, candidate_games=40, query_games=10):
    """Enforce complete fixed candidate/query counts with one question per player."""
    labeled = labeled_queries(queries, truth)
    require(candidates.player_id.nunique() == players, 'Wrong FINAL TEST 2 player count')
    require((candidates.groupby('player_id').size() == candidate_games).all(), 'Wrong candidate games/player')
    require(len(truth) == players and not truth.player_id.duplicated().any(), 'Need one query question per player')
    require(set(truth.player_id) == set(candidates.player_id), 'Candidate/query identities differ')
    require((queries.groupby('question_id').size() == query_games).all(), 'Wrong query games/question')
    return labeled


def overlap_audit(partitions):
    """Reject duplicate games and cross-partition game/SGF or identity leakage."""
    output = {}
    for name, frame in partitions.items():
        require(not frame.game_id.duplicated().any(), f'{name}: duplicate game_id')
        require(not frame.sgf_content.duplicated().any(), f'{name}: duplicate SGF')
    for (left, a), (right, b) in itertools.combinations(partitions.items(), 2):
        counts = {key: len(set(a[key]) & set(b[key])) for key in ['player_id', 'game_id', 'sgf_content']}
        # Candidate and query identities must match; all other partitions must be disjoint.
        require(counts['game_id'] == 0 and counts['sgf_content'] == 0, f'{left}/{right}: game or SGF overlap')
        if {left, right} not in [{'final_candidate', 'final_query'}, {'val_candidate', 'val_query'}]:
            require(counts['player_id'] == 0, f'{left}/{right}: player overlap')
        output[f'{left}/{right}'] = counts
    return output


def preflight(config_path=None):
    """Verify immutable artifacts, complete source provenance, and zero evaluations."""
    require_unconsumed()
    path = resolve_path(config_path or 'configs/phase26_selected.yaml')
    require(path == ROOT / 'configs/phase26_selected.yaml', 'Only the frozen selected config is allowed')
    require(path.exists(), 'Missing phase26_selected.yaml')
    config = load_config(path)
    choice = config['selection']
    require(choice.get('frozen') is True, 'Selection is not frozen')
    require(choice['selected_experiment'] == config['experiment'] == 'H20-Hard', 'Wrong selected experiment')
    require(choice['best_epoch'] == 17, 'Wrong selected epoch')
    require(abs(choice['dev_val_score'] - 0.37670072252240877) < 1e-12, 'Wrong frozen DEV score')
    require(config['seed'] == 42 and config['inference']['top_k'] == 5, 'Seed/Top-k changed')
    require(config['features'] == {'board_size': 19, 'history_length': 8, 'max_positions_per_game': 16}, 'Features changed')
    require(config['model'] == {'in_channels': 17, 'channels': 64, 'num_blocks': 8,
                               'embedding_dim': 128, 'pool_size': 3}, 'Model changed')
    require(config['training']['triplet_strategy'] == 'batch_hard', 'Triplet strategy changed')
    require(config['split']['test_players'] == 50 and config['split']['test_candidate_games'] == 40
            and config['split']['test_query_games'] == 10, 'Final split configuration changed')
    frozen = read_json('outputs/phase26/selected/selection_frozen.json')
    state = read_json(config['paths']['training_state_json'])
    summary = read_json('outputs/phase26/summary.json')
    require(frozen['final_test2_evaluations'] == summary['final_test2_evaluations'] == state['test_evaluations'] == 0,
            'FINAL TEST 2 evaluation count is not zero')
    require(frozen['config_sha256'] == file_digest(path), 'Frozen config SHA256 mismatch')
    require(choice['experiment_signature'] == experiment_signature(config), 'Frozen hyperparameters changed')
    checkpoint_path = resolve_path(config['paths']['best_checkpoint'])
    require(checkpoint_path == ROOT / 'outputs/phase26/selected/best.pt', 'Wrong checkpoint path')
    checkpoint_hash = file_digest(checkpoint_path)
    require(checkpoint_hash == choice['best_checkpoint_sha256'] == frozen['best_checkpoint_sha256']
            == state['best_checkpoint_sha256'], 'Checkpoint SHA256 mismatch')
    require(state['status'] == 'completed' and state['best_epoch'] == 17
            and state['best_validation_score'] == choice['dev_val_score'], 'Training state mismatch')
    for key, digest in choice['dev_partition_sha256'].items():
        require(file_digest(config['paths'][key]) == digest, f'Frozen DEV input changed: {key}')
    master = read_json('outputs/phase26/splits/split_audit.json')
    require(master['seed'] == 42 and master['train_player_count'] == 200 and master['val_player_count'] == 50
            and master['test_player_count'] == 50, 'Master split changed')
    source_hash = file_digest(config['paths']['training_csv'])
    require(source_hash == master['source_csv_sha256'], 'Official dataset SHA256 changed')
    original_paths = {'metric_train_csv': 'outputs/phase26/splits/dev_train.csv',
                      'val_candidates_csv': 'outputs/phase26/splits/dev_val_candidates.csv',
                      'val_queries_csv': 'outputs/phase26/splits/dev_val_queries.csv',
                      'val_ground_truth_csv': 'outputs/phase26/splits/dev_val_ground_truth.csv',
                      'test_candidates_csv': 'outputs/phase26/splits/final_test2_candidates.csv',
                      'test_queries_csv': 'outputs/phase26/splits/final_test2_queries.csv',
                      'test_ground_truth_csv': 'outputs/phase26/splits/final_test2_ground_truth.csv'}
    hashes = {}
    for key, source_path in original_paths.items():
        if key != 'metric_train_csv':
            require(resolve_path(config['paths'][key]) == resolve_path(source_path), f'Wrong split path: {key}')
        hashes[key] = file_digest(source_path)
        require(hashes[key] == master['partition_sha256'][key], f'Master split SHA256 mismatch: {key}')
    required = ['player_id', 'game_id', 'color', 'sgf_content']
    train = read_csv(original_paths['metric_train_csv'], required)
    vc = read_csv(original_paths['val_candidates_csv'], required)
    vq = labeled_queries(read_csv(original_paths['val_queries_csv'], ['question_id', 'game_id', 'color', 'sgf_content']),
                         read_csv(original_paths['val_ground_truth_csv'], ['question_id', 'player_id']))
    candidates = read_csv(original_paths['test_candidates_csv'], required)
    queries = read_csv(original_paths['test_queries_csv'], ['question_id', 'game_id', 'color', 'sgf_content'])
    truth = read_csv(original_paths['test_ground_truth_csv'], ['question_id', 'player_id'])
    fq = check_final_counts(candidates, queries, truth)
    closed = pd.concat([read_csv('outputs/splits/test_candidates.csv', required),
        labeled_queries(read_csv('outputs/splits/test_queries.csv', ['question_id', 'game_id', 'color', 'sgf_content']),
                        read_csv('outputs/splits/test_ground_truth.csv', ['question_id', 'player_id']))], ignore_index=True)
    require(train.player_id.nunique() == 200 and len(train) == 4000
            and vc.player_id.nunique() == 50 and len(vc) == 1000 and len(vq) == 500, 'Master partition counts changed')
    selected_train = read_csv(config['paths']['metric_train_csv'], required)
    require(set(selected_train.game_id) <= set(train.game_id), 'Selected training subset not in master TRAIN')
    partitions = {'train': train, 'val_candidate': vc, 'val_query': vq,
                  'final_candidate': candidates, 'final_query': fq, 'round1_closed': closed}
    overlaps = overlap_audit(partitions)
    official = read_csv(config['paths']['training_csv'], required + ['rank']).set_index('game_id', verify_integrity=True)
    for name, frame in partitions.items():
        require(set(frame.game_id) <= set(official.index), f'{name}: unknown source games')
        originals = official.loc[frame.game_id].reset_index(drop=True)
        for column in ['player_id', 'color', 'sgf_content']:
            require(originals[column].equals(frame[column].reset_index(drop=True)), f'{name}: source {column} mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(checkpoint['epoch'] == 17 and checkpoint['experiment'] == 'H20-Hard', 'Checkpoint identity/epoch mismatch')
    require(checkpoint['model_config'] == config['model'] and checkpoint['feature_config'] == config['features']
            and checkpoint['feature_seed'] == 42 and checkpoint['training_config'] == config['training'], 'Checkpoint settings mismatch')
    require(checkpoint['train_source_sha256'] == file_digest(config['paths']['metric_train_csv']), 'Checkpoint TRAIN mismatch')
    preserve_round1(config)
    report = {'status': 'PASSED', 'evaluation_count': 0, 'dataset_sha256': source_hash,
              'split_sha256': hashes, 'master_split_audit_sha256': file_digest('outputs/phase26/splits/split_audit.json'),
              'checkpoint_sha256': checkpoint_hash, 'config_sha256': file_digest(path),
              'selected_experiment': 'H20-Hard', 'best_epoch': 17, 'overlaps': overlaps,
              'final_players': 50, 'candidate_games': 2000, 'query_games': 500}
    return config, candidates, queries, truth, report
