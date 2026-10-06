"""Independent stability validation: immutable methods, paired bootstrap, no selection."""
import contextvars
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from .metric_utils import choose_device, file_digest, seed_everything, write_json
from .phase28_common import encoder, fuse, metrics, rankings
from .phase28_opening import opening_scores
from .phase28_triplet import score_embeddings, diagnostics
from .run_phase28 import cached_store
from .utils import ROOT, load_config, read_csv, write_csv

_active = contextvars.ContextVar('phase29_stability_only', default=False)
FIXED = {'opening_mode': 'player', 'opening_window': 5, 'color_aware': True,
         'triplet_experiment': 'Triplet-Hard-100', 'best_epoch': 18, 'alpha': .9, 'normalization': 'z-score'}
CHECKPOINT = 'outputs/phase28/experiments/Triplet-Hard-100/best.pt'


def _guard(event, args):
    """Refuse closed TEST files, DEV2 raw inputs and writes to all Phase 2.8 artifacts."""
    if not _active.get() or event != 'open' or not isinstance(args[0], (str, bytes, Path)):
        return
    path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve().as_posix().lower()
    if ('/outputs/splits/test_' in path or '/outputs/round1_archive/' in path
        or '/outputs/phase26/' in path):
        raise PermissionError('Phase 2.9 cannot read CLOSED TEST files, including saved results')
    if '/outputs/phase28/splits/' in path and not path.endswith(('/audit.json', '/detailed_audit.json')):
        raise PermissionError('Phase 2.9 cannot read DEV2 raw splits or truths')
    if '/outputs/phase28/features/' in path or ('/outputs/phase28/experiments/' in path and not path.endswith('/triplet-hard-100/best.pt')):
        raise PermissionError('Phase 2.9 cannot search or evaluate other DEV2 checkpoints')
    if '/outputs/phase28/' in path or path.endswith('/configs/phase28_selected.yaml'):
        import os
        if any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0) & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
            raise PermissionError('Phase 2.8 artifacts are read-only')


sys.addaudithook(_guard)


@contextmanager
def stability_only():
    """Protect every preparation, inference and reporting step from old TEST access."""
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


def validate_fixed(config, selected):
    """Reject any window, alpha, checkpoint or epoch substitution before data access."""
    if config['fixed'] != FIXED:
        raise ValueError('Phase 2.9 fixed method protocol changed; selection/search forbidden')
    s = selected['selection']
    if not s['frozen'] or s['kind'] != 'fusion' or s['alpha'] != .9 or s['fusion_normalization'] != 'z-score':
        raise ValueError('Phase 2.8 fusion is not the frozen alpha=0.9 z-score selection')
    if s['opening'] != {'method': 'Color-aware-player-5', 'mode': 'player', 'window': 5, 'color_aware': True}:
        raise ValueError('Opening must stay player-5 color-aware')
    t = s['triplet']
    if t['experiment'] != 'Triplet-Hard-100' or t['best_epoch'] != 18 or t['checkpoint'] != CHECKPOINT:
        raise ValueError('Checkpoint/epoch selection is forbidden')
    if config['split'] != {'players': 100, 'candidate_games': 30, 'query_games': 10}:
        raise ValueError('Stability split budget must remain 100/30/10')
    if config['bootstrap'] != {'resamples': 1000, 'seed': 42, 'confidence': .95}:
        raise ValueError('Bootstrap protocol changed')


def exclusion_metadata():
    """Read previously saved exclusion/DEV2 identity lists without reopening old inputs."""
    audit = json.loads((ROOT/'outputs/phase28/splits/audit.json').read_text(encoding='utf-8'))
    detail = json.loads((ROOT/'outputs/phase28/splits/detailed_audit.json').read_text(encoding='utf-8'))
    groups = {key: set(audit['excluded_provenance'][key]) for key in ['round1_test', 'phase26_val', 'final_test2']}
    groups.update(dev2_train=set(detail['train_player_ids']), dev2_val=set(detail['val_player_ids']))
    expected = dict(zip(groups, [10, 50, 50, 200, 100]))
    if any(len(groups[k]) != expected[k] for k in groups):
        raise ValueError('Exclusion metadata player counts changed')
    excluded = set().union(*groups.values())
    if len(excluded) != 410:
        raise ValueError('Historical identity sets must be disjoint')
    return excluded, {key: sorted(ids) for key, ids in groups.items()}


def make_stability(source, excluded, settings, seed):
    """Draw 100 unseen identities once; insufficient eligibility reports the maximum and stops."""
    if source.game_id.duplicated().any() or source.sgf_content.duplicated().any():
        raise ValueError('Source game ID/SGF is not globally unique')
    available = source[~source.player_id.isin(excluded)]
    counts = available.groupby('player_id').size()
    eligible = sorted(counts[counts >= settings['candidate_games']+settings['query_games']].index)
    if len(eligible) < settings['players']:
        raise ValueError(f'Insufficient Stability players: maximum eligible={len(eligible)}, requested={settings["players"]}; no reduction performed')
    rng = np.random.default_rng(seed)
    ids = sorted(rng.permutation(eligible)[:settings['players']].tolist())
    cs, qs, ts = [], [], []
    for i, player in enumerate(ids):
        games = available[available.player_id == player].sort_values('game_id')
        games = games.iloc[rng.permutation(len(games))]
        n, m, question = settings['candidate_games'], settings['query_games'], f'stability_q_{i:04d}'
        cs.append(games.iloc[:n])
        qs.append(games.iloc[n:n+m][['game_id', 'color', 'sgf_content']].assign(question_id=question))
        ts.append({'question_id': question, 'player_id': player})
    c, q, t = pd.concat(cs, ignore_index=True), pd.concat(qs, ignore_index=True), pd.DataFrame(ts)
    audit = {'players': len(ids), 'max_eligible_players': len(eligible), 'player_ids': ids,
        'candidate_games': len(c), 'query_games': len(q), 'excluded_player_overlap': len(set(ids)&excluded),
        'game_id_overlap': len(set(c.game_id)&set(q.game_id)), 'sgf_content_overlap': len(set(c.sgf_content)&set(q.sgf_content)),
        'source_game_id_and_exact_sgf_globally_unique': True}
    if any(audit[k] for k in ['excluded_player_overlap', 'game_id_overlap', 'sgf_content_overlap']):
        raise ValueError('Stability data leakage')
    return c, q, t, audit


def prepare(config, selected_sha):
    """Create or hash-verify a fixed new split; never redraw after observing scores."""
    excluded, groups = exclusion_metadata()
    directory = ROOT/config['output_dir']/'splits'
    audit_path = directory/'audit.json'
    source_sha = file_digest(config['paths']['training_csv'])
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if any(audit[k] != value for k, value in [('source_sha256', source_sha), ('seed', config['seed']),
            ('settings', config['split']), ('exclusions', groups), ('selected_config_sha256', selected_sha)]):
            raise ValueError('Existing Stability split provenance changed; refusing redraw')
        for name, sha in audit['file_sha256'].items():
            if file_digest(directory/name) != sha:
                raise ValueError('Stability split changed')
        return audit
    source = read_csv(config['paths']['training_csv'], ['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
    if not excluded <= set(source.player_id) or not set(source.color) <= {'B', 'W'}:
        raise ValueError('Unknown exclusion ID or color')
    c, q, t, audit = make_stability(source, excluded, config['split'], config['seed'])
    for name, frame in [('candidates.csv', c), ('queries.csv', q), ('ground_truth.csv', t)]:
        write_csv(frame, directory/name)
    audit.update(seed=config['seed'], settings=config['split'], exclusions=groups, source_sha256=source_sha,
        selected_config_sha256=selected_sha, file_sha256={name: file_digest(directory/name) for name in ['candidates.csv', 'queries.csv', 'ground_truth.csv']})
    write_json(audit, audit_path)
    return audit


def paired_bootstrap(opening, fusion, resamples=1000, seed=42):
    """Resample paired questions, preserving covariance in Fusion minus Opening."""
    if opening.question_id.duplicated().any() or fusion.question_id.duplicated().any() or set(opening.question_id) != set(fusion.question_id):
        raise ValueError('Opening/Fusion must have identical unique questions')
    a = opening.sort_values('question_id').competition_score.to_numpy(dtype=float)
    b = fusion.sort_values('question_id').competition_score.to_numpy(dtype=float)
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all() or resamples < 1:
        raise ValueError('Invalid question scores/bootstrap budget')
    indices = np.random.default_rng(seed).integers(len(a), size=(resamples, len(a)))
    samples = {'opening': a[indices].mean(axis=1), 'fusion': b[indices].mean(axis=1),
               'fusion_minus_opening': (b-a)[indices].mean(axis=1)}
    original = {'opening': a.mean(), 'fusion': b.mean(), 'fusion_minus_opening': (b-a).mean()}
    rows = [{'method': key, 'mean': float(original[key]), 'bootstrap_mean': float(values.mean()),
        'ci_lower': float(np.percentile(values, 2.5)), 'ci_upper': float(np.percentile(values, 97.5)),
        'resamples': resamples, 'seed': seed} for key, values in samples.items()]
    return rows, pd.DataFrame(samples)


def readiness(opening_score, fusion_score, boot_mean, players, opening_ci_lower):
    """Apply the requested three conditions; do not turn CI into an extra selection rule."""
    # Uniform random ranking expected score, used as an analytic reference only.
    random_expected = float(sum(np.exp(-np.arange(min(5, players))))/players)
    effective = opening_ci_lower > random_expected
    ready = effective and fusion_score > opening_score and boot_mean > 0
    return {'status': 'READY FOR FINAL TEST 3' if ready else 'NOT READY FOR FINAL TEST 3',
            'opening_effective': effective, 'analytic_random_expected_score': random_expected,
            'fusion_above_opening': fusion_score > opening_score, 'bootstrap_mean_positive': boot_mean > 0,
            'effectiveness_definition': 'Opening 95% bootstrap CI lower bound exceeds analytic uniform-random expected score'}


def run(config):
    """Load exactly one frozen checkpoint and infer three fixed methods on unseen players."""
    started = time.perf_counter()
    with stability_only():
        selected = load_config(ROOT/config['selected_config'])
        validate_fixed(config, selected)
        protocol = json.loads((ROOT/config['protocol_manifest']).read_text(encoding='utf-8'))
        selected_sha = file_digest(config['selected_config'])
        if selected_sha != protocol['code_config_sha256']['configs/phase28_selected.yaml']:
            raise ValueError('Frozen Phase 2.8 config SHA256 mismatch')
        for code in ['src/player_model.py', 'src/phase28_common.py', 'src/phase28_triplet.py']:
            if file_digest(code) != protocol['code_config_sha256'][code]:
                raise ValueError('Frozen encoder/inference code changed')
        if file_digest(config['paths']['training_csv']) != selected['selection']['source_sha256']:
            raise ValueError('Official source changed since frozen Phase 2.8')
        checkpoint_sha = file_digest(CHECKPOINT)
        if checkpoint_sha != selected['selection']['triplet']['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint SHA256 mismatch')
        directory = ROOT/config['output_dir']
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/'summary.json').exists():
            raise ValueError('Stability validation completed; use saved results, no repeated inference')
        audit = prepare(config, selected_sha)
        c, q, t = [read_csv(config['paths'][key], columns) for key, columns in [
            ('val_candidates_csv', ['player_id', 'game_id', 'color', 'sgf_content']),
            ('val_queries_csv', ['question_id', 'game_id', 'color', 'sgf_content']),
            ('val_ground_truth_csv', ['question_id', 'player_id'])]]
        oscores, players, questions, errors, missing = opening_scores(c, q, 'player', 5, True)
        runtime = copy.deepcopy(selected)
        runtime['paths'] = config['paths']
        runtime['cache'] = selected['cache']
        cs, cc = cached_store(runtime, 'val_candidate', c, 'player_id', 'val_candidates_csv', 'val_candidate_manifest')
        qs, qc = cached_store(runtime, 'val_query', q, 'question_id', 'val_queries_csv', 'val_query_manifest')
        write_csv(pd.DataFrame([cc, qc]), directory/'feature_coverage.csv')
        seed_everything(config['seed'], True, selected['training']['num_threads'])
        device = choose_device(config['inference']['device'])
        model = encoder(selected, device)
        checkpoint = torch.load(ROOT/CHECKPOINT, map_location='cpu', weights_only=True)
        if checkpoint['epoch'] != 18 or checkpoint['model_config'] != selected['model'] or checkpoint['feature_config'] != selected['features']:
            raise ValueError('Checkpoint epoch/architecture/features mismatch')
        if not set(checkpoint['training_players']) <= set(audit['exclusions']['dev2_train']):
            raise ValueError('Selected checkpoint TRAIN IDs mismatch')
        model.load_state_dict(checkpoint['model_state_dict'])
        tscores, cg, qg = score_embeddings(model, cs, qs, device, config['inference']['batch_size'], players, questions)
        fscores = fuse(oscores, tscores, .9, 'z-score')
        values, details = {}, {}
        for name, scores in [('opening', oscores), ('triplet', tscores), ('fusion', fscores)]:
            values[name], details[name] = metrics(scores, players, questions, t)
            write_csv(rankings(scores, players, questions), directory/(name+'_predictions.csv'))
            write_csv(details[name], directory/(name+'_question_scores.csv'))
        np.savez_compressed(directory/'complete_scores.npz', opening=oscores, triplet=tscores, fusion=fscores,
                            players=np.array(players), questions=np.array(questions))
        bootstrap, samples = paired_bootstrap(details['opening'], details['fusion'], **{k: config['bootstrap'][k] for k in ['resamples', 'seed']})
        write_csv(pd.DataFrame(bootstrap), directory/'bootstrap_ci.csv')
        write_csv(samples, directory/'bootstrap_samples.csv')
        diag, variance = diagnostics(cg, qg, cs.records, qs.records, t, selected['diagnostics'])
        write_csv(pd.DataFrame([diag]), directory/'embedding_diagnostics.csv')
        write_csv(pd.DataFrame({'dimension': np.arange(len(variance)), 'variance': variance}), directory/'dimension_variance.csv')
        # Saved DEV2 metrics are a fixed descriptive reference, never a search input.
        previous = json.loads((ROOT/'outputs/phase28/summary.json').read_text(encoding='utf-8'))
        refs = {'opening': previous['opening_best'], 'triplet': previous['triplet_best'], 'fusion': previous['selected']}
        if refs['opening']['method'] != 'Color-aware-player-5' or refs['triplet']['experiment'] != 'Triplet-Hard-100' or refs['fusion']['alpha'] != .9:
            raise ValueError('Historical descriptive references no longer match frozen methods')
        comparison = [{'method': name, 'dev2_score': refs[name]['competition_score'],
            'stability_score': values[name]['competition_score'], 'difference': values[name]['competition_score']-refs[name]['competition_score'],
            **{f'{side}_{metric}': data[metric] for side, data in [('dev2', refs[name]), ('stability', values[name])] for metric in ['top1', 'top3', 'top5']}}
            for name in values]
        write_csv(pd.DataFrame(comparison), directory/'dev2_vs_stability.csv')
        decision = readiness(values['opening']['competition_score'], values['fusion']['competition_score'], bootstrap[2]['bootstrap_mean'], len(players), bootstrap[0]['ci_lower'])
        if file_digest(CHECKPOINT) != checkpoint_sha or file_digest(config['selected_config']) != selected_sha:
            raise ValueError('Frozen artifact changed during inference')
        summary = {'status': 'COMPLETE STABILITY VALIDATION', 'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.perf_counter()-started, 'split': audit, 'fixed': FIXED,
            'selected_config_sha256': selected_sha, 'checkpoint_sha256': checkpoint_sha, 'metrics': values,
            'comparison': comparison, 'bootstrap': bootstrap, 'embedding': diag, 'feature_coverage': [cc, qc],
            'opening_failed_games': len(errors), 'missing_candidate_colors': missing, 'decision': decision,
            'inference_only': True, 'closed_test_accessed': False, 'dev2_used_for_selection': False, 'final_test3_created': False}
        write_csv(pd.DataFrame(errors, columns=['group_id', 'game_id', 'error']), directory/'opening_errors.csv')
        write_json(summary, directory/'summary.json')
        print(json.dumps({'metrics': values, 'bootstrap': bootstrap, 'decision': decision}, indent=2), flush=True)
        return summary


def main():
    """Run the immutable stability protocol with a configurable device only."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/phase29.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    directory = ROOT/config['output_dir']
    directory.mkdir(parents=True, exist_ok=True)
    class Tee:
        """Preserve complete UTF-8 inference output while showing live progress."""
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
    stdout, stderr = sys.stdout, sys.stderr
    with (directory/'run.log').open('a', encoding='utf-8') as stream:
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try:
            run(config)
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == '__main__':
    main()
