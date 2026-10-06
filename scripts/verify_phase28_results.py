"""Audit completed DEV2 results, exposure and selected provenance without inference."""
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metric_utils import file_digest, write_json
from src.phase28_common import dev2_only


def require(condition, message):
    """Keep result-audit assertions active even under python -O."""
    if not condition:
        raise ValueError(message)


def main():
    """Verify every completed epoch and matched cohort, then snapshot result hashes."""
    directory = ROOT/'outputs/phase28'
    with dev2_only():
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        require(summary['status'] == 'COMPLETE DEV2 ONLY', 'Incomplete DEV2 run')
        split = json.loads((directory/'splits/audit.json').read_text(encoding='utf-8'))
        for name, digest in split['file_sha256'].items():
            require(file_digest(directory/'splits'/name) == digest, 'Split changed')
        train = pd.read_csv(directory/'splits/train.csv', dtype=str)
        candidates = pd.read_csv(directory/'splits/val_candidates.csv', dtype=str)
        truth = pd.read_csv(directory/'splits/val_ground_truth.csv', dtype=str)
        queries = pd.read_csv(directory/'splits/val_queries.csv', dtype=str)
        require('player_id' not in queries and 'rank' not in queries, 'Query identity metadata leak')
        require(not set(train.player_id) & set(candidates.player_id), 'TRAIN/VAL identity overlap')
        require(set(candidates.player_id) == set(truth.player_id), 'VAL pool changed')
        exposure = pd.read_csv(directory/'exposure_results.csv')
        cross = pd.read_csv(directory/'cross_color_triplet_results.csv')
        all_runs = pd.concat([exposure, cross]).drop_duplicates('experiment')
        audited = []
        for _, row in all_runs.iterrows():
            target = directory/'experiments'/row.experiment
            state = json.loads((target/'state.json').read_text(encoding='utf-8'))
            history = pd.read_csv(target/'training_log.csv')
            require(len(history) == 20 and history.epoch.tolist() == list(range(1, 21)), 'Epoch protocol changed')
            require((history.samples_per_epoch == len(state['players'])*20).all(), 'Exposure total changed')
            require((history.min_anchor_exposure == 20).all() and (history.max_anchor_exposure == 20).all(), 'Unequal player exposure')
            best = history.iloc[history.competition_score.argmax()]
            require(int(best.epoch) == state['best_epoch'] == int(row.best_epoch), 'Best epoch not selected by DEV2 score')
            require(abs(best.competition_score-row.competition_score) < 1e-12, 'Saved score mismatch')
            require(file_digest(target/'best.pt') == state['checkpoint_sha256'], 'Checkpoint hash changed')
            checkpoint = torch.load(target/'best.pt', map_location='cpu', weights_only=True)
            require(checkpoint['epoch'] == state['best_epoch'], 'Checkpoint epoch mismatch')
            require(checkpoint['training_players'] == state['players'], 'Checkpoint cohort mismatch')
            require(set(train[train.player_id.isin(state['players'])].groupby('player_id').size()) == {20}, 'Training games budget changed')
            audited.append({'experiment': row.experiment, 'players': len(state['players']),
                'epochs': len(history), 'best_epoch': state['best_epoch'], 'checkpoint_sha256': state['checkpoint_sha256'],
                'signature': state['signature'], 'anchor_exposure_min': 20, 'anchor_exposure_max': 20})
        matched, cross_row = cross.experiment.tolist()
        regular = json.loads((directory/'experiments'/matched/'state.json').read_text(encoding='utf-8'))
        opposite = json.loads((directory/'experiments'/cross_row/'state.json').read_text(encoding='utf-8'))
        require(regular['players'] == opposite['players'], 'CrossColor unmatched training cohort')
        require(not regular['cross_color_positive'] and opposite['cross_color_positive'], 'Positive strategy changed')
        colors = train.groupby(['player_id', 'color']).size().unstack(fill_value=0)
        require((colors.loc[opposite['players'], ['B', 'W']] >= 2).all().all(), 'CrossColor eligibility failure')
        control_audit = json.loads((directory/'color_controls/audit.json').read_text(encoding='utf-8'))
        for name in control_audit['conditions']:
            target = directory/'color_controls'/name
            c, q, t = [pd.read_csv(target/(part+'.csv'), dtype=str) for part in ['candidates', 'queries', 'ground_truth']]
            require(sorted(c.player_id.unique()) == sorted(t.player_id) == control_audit['player_ids'], 'Color cohort composition changed')
            require(set(c.groupby('player_id').size()) == {30} and set(q.groupby('question_id').size()) == {10}, 'Color game budgets differ')
            require(not set(c.game_id)&set(q.game_id) and not set(c.sgf_content)&set(q.sgf_content), 'Color game leakage')
        selected = yaml.safe_load((ROOT/'configs/phase28_selected.yaml').read_text(encoding='utf-8'))
        opening = pd.read_csv(directory/'opening_results.csv')
        fusion = pd.read_csv(directory/'fusion_results.csv')
        require(fusion.alpha.tolist() == [i/10 for i in range(11)], 'Fusion grid changed')
        maximum = max(opening.competition_score.max(), all_runs.competition_score.max(), fusion.competition_score.max())
        require(selected['selection']['frozen'] and abs(selected['selection']['competition_score']-maximum) < 1e-12, 'Selection not frozen at max DEV2 score')
        source_files = [ROOT/'configs/phase28.yaml', ROOT/'configs/phase28_selected.yaml', ROOT/'src/player_model.py']
        source_files.extend(sorted((ROOT/'src').glob('phase28_*.py')))
        source_files.append(ROOT/'src/run_phase28.py')
        snapshot = {'status': 'VERIFIED DEV2 ONLY', 'experiments': audited,
            'matched_cross_color_training_ids': True, 'color_control_players': len(control_audit['player_ids']),
            'identical_color_control_ids_and_game_budgets': True, 'selection_uses_only_dev2_score': True,
            'code_config_sha256': {p.relative_to(ROOT).as_posix(): file_digest(p) for p in source_files},
            'result_sha256': {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(directory.glob('*.csv'))},
            'closed_test_truth_or_inference_used': False, 'final_test3_created': False}
        write_json(snapshot, directory/'protocol_verification.json')
    print('Phase 2.8 results verified: exposure, epochs, SHA256, matched colors, DEV2-only selection')


if __name__ == '__main__':
    main()
