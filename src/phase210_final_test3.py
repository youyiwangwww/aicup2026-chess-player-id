"""Preregister and consume a genuinely unseen FINAL TEST 3 exactly once."""
import argparse
import copy
import json
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
import torch

from . import final_test3_lock as lock
from .metric_utils import choose_device, file_digest, seed_everything, write_json
from .phase28_common import encoder, fuse, metrics, rankings, dev2_only
from .phase28_opening import opening_scores
from .phase28_triplet import score_embeddings
from .phase29_validation import CHECKPOINT, validate_fixed, make_stability, paired_bootstrap
from .run_phase28 import cached_store
from .utils import ROOT, load_config, read_csv, write_csv

NAMES = ['random', 'opening_color_player5', 'triplet_hard100', 'fusion_alpha09']


def read_json(path):
    """Read only registration or allowed saved summaries, never historical TEST results."""
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def validate_config(config, selected):
    """Reject altered methods, seeds, features, architecture, paths or inference budgets."""
    validate_fixed(config, selected)
    expected = load_config(ROOT/'configs/phase210.yaml')
    if config != expected or config['seed'] != 42 or config['output_dir'] != 'outputs/phase210':
        raise ValueError('Only the preregistered FINAL TEST 3 configuration is allowed')
    if selected['features']['history_length'] != 8 or selected['features']['max_positions_per_game'] != 16:
        raise ValueError('Frozen features must be history 8 / positions 16')
    if any(selected['model'][key] != value for key, value in [('channels',64), ('num_blocks',8), ('embedding_dim',128)]):
        raise ValueError('Frozen architecture changed')


def registration(config):
    """Create one immutable split and protocol before any inference or truth parsing."""
    lock.require_unconsumed()
    selected = load_config(ROOT/config['selected_config'])
    validate_config(config, selected)
    directory = lock.directory()
    protocol_path = directory/'preregistered_protocol.json'
    if protocol_path.exists():
        return preflight(config)[0]
    eligibility = read_json('outputs/phase210/eligibility.json')
    if eligibility['maximum_eligible_unseen_players'] < 100:
        raise ValueError(f"maximum eligible unseen players = {eligibility['maximum_eligible_unseen_players']}; TEST not created")
    for path, digest in eligibility['identity_source_sha256'].items():
        if file_digest(path) != digest:
            raise ValueError('Historical TRAIN identity source changed')
    for path in ['outputs/phase29/splits/audit.json', 'outputs/phase28/splits/audit.json', 'outputs/phase28/splits/detailed_audit.json']:
        # Bind the copied metadata containing old TEST identities; no old TEST file is opened.
        eligibility['identity_source_sha256'][path] = file_digest(path)
    excluded = set().union(*(set(ids) for ids in eligibility['groups'].values()))
    source = read_csv(config['paths']['training_csv'], ['player_id','game_id','rank','color','sgf_content'])
    if file_digest(config['paths']['training_csv']) != selected['selection']['source_sha256']:
        raise ValueError('Official source changed')
    if not excluded <= set(source.player_id) or not set(source.color) <= {'B','W'}:
        raise ValueError('Unknown historical identity/color')
    c, q, truth, audit = make_stability(source, excluded, config['split'], 42)
    q['question_id'] = q.question_id.str.replace('stability_q_', 'final3_q_', regex=False)
    truth['question_id'] = truth.question_id.str.replace('stability_q_', 'final3_q_', regex=False)
    audit.update(unseen_players=eligibility['unseen_players'], historical_unique_players=len(excluded),
                 historical_identity_intersections={key: len(set(audit['player_ids'])&set(ids)) for key,ids in eligibility['groups'].items()},
                 question_player_mapping=dict(zip(truth.question_id,truth.player_id)), seed=42)
    with lock.stage('PREPARATION'):
        for name, frame in [('candidates.csv',c), ('queries.csv',q), ('ground_truth.csv',truth)]:
            path = directory/'splits'/name
            if path.exists():
                import hashlib
                expected = hashlib.sha256(frame.to_csv(index=False).encode('utf-8-sig')).hexdigest()
                if lock.input_digest(path) != expected:
                    raise ValueError('Partial preparation differs; refusing replacement or redraw')
            else:
                write_csv(frame, path)
        write_json(audit, directory/'splits/split_audit.json')
        bindings = {str(path.relative_to(ROOT)).replace('\\','/'): lock.input_digest(path) for path in
                    [directory/'splits'/name for name in ['candidates.csv','queries.csv','ground_truth.csv','split_audit.json']]}
        paths = [config['paths']['training_csv'], config['selected_config'], CHECKPOINT, 'configs/phase210.yaml',
                 'src/player_model.py','src/phase28_common.py','src/phase28_triplet.py','src/phase28_opening.py',
                 'src/phase210_final_test3.py','src/final_test3_lock.py','outputs/phase210/eligibility.json']
        bindings.update({path: file_digest(path) for path in paths})
        bindings.update(eligibility['identity_source_sha256'])
        protocol = {'status':'PREREGISTERED','evaluation_count':0,'timestamp':datetime.now(timezone.utc).isoformat(),
                    'bindings':bindings,'fixed':config['fixed'],'split':config['split'],'seed':42,
                    'bootstrap':config['bootstrap'],'ground_truth_rule':'parse only after all four predictions are saved'}
        with protocol_path.open('x',encoding='utf-8') as stream:
            json.dump(protocol, stream, indent=2, ensure_ascii=False)
        with (directory/'preregistered_protocol.sha256').open('x',encoding='utf-8') as stream:
            stream.write(file_digest(protocol_path)+'\n')
    preflight(config)
    return protocol


def preflight(config):
    """Verify all 13 gates without parsing ground truth or performing model inference."""
    lock.require_unconsumed()
    selected = load_config(ROOT/config['selected_config'])
    validate_config(config,selected)
    directory=lock.directory()
    protocol=read_json('outputs/phase210/preregistered_protocol.json')
    if file_digest(directory/'preregistered_protocol.json') != (directory/'preregistered_protocol.sha256').read_text().strip():
        raise ValueError('Protocol SHA256 mismatch')
    if protocol['status'] != 'PREREGISTERED' or protocol['evaluation_count'] != 0 or protocol['fixed'] != config['fixed'] or protocol['split'] != config['split']:
        raise ValueError('Protocol settings/count mismatch')
    with lock.stage('PREFLIGHT'):
        for path,digest in protocol['bindings'].items():
            if lock.input_digest(path) != digest:
                raise ValueError(f'Preregistered SHA256 mismatch: {path}')
    old_protocol=read_json('outputs/phase28/protocol_verification.json')
    for path in ['configs/phase28_selected.yaml','src/player_model.py','src/phase28_common.py','src/phase28_triplet.py']:
        if file_digest(path) != old_protocol['code_config_sha256'][path]:
            raise ValueError('Frozen model/config code SHA256 mismatch')
    if file_digest(CHECKPOINT) != selected['selection']['triplet']['checkpoint_sha256']:
        raise ValueError('Selected checkpoint SHA256 mismatch')
    checkpoint=torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=True)
    if checkpoint['epoch'] != 18 or checkpoint['model_config'] != selected['model'] or checkpoint['feature_config'] != selected['features']:
        raise ValueError('Checkpoint epoch/model/features mismatch')
    audit=read_json('outputs/phase210/splits/split_audit.json')
    eligibility=read_json('outputs/phase210/eligibility.json')
    if not set(checkpoint['training_players']) <= set(eligibility['groups']['dev2_train']):
        raise ValueError('Checkpoint training identity provenance mismatch')
    c=read_csv(config['paths']['val_candidates_csv'],['player_id','game_id','color','sgf_content'])
    q=read_csv(config['paths']['val_queries_csv'],['question_id','game_id','color','sgf_content'])
    ids=set(c.player_id)
    historical=set().union(*(set(v) for v in eligibility['groups'].values()))
    if len(ids)!=100 or ids & historical or any(audit['historical_identity_intersections'].values()):
        raise ValueError('FINAL TEST 3 identities are not new/disjoint')
    if len(c)!=3000 or len(q)!=1000 or not c.groupby('player_id').size().eq(30).all() or len(q.question_id.unique())!=100 or not q.groupby('question_id').size().eq(10).all():
        raise ValueError('FINAL TEST 3 budget not 100/30/10')
    if {'player_id','rank'} & set(q.columns) or set(c.game_id)&set(q.game_id) or set(c.sgf_content)&set(q.sgf_content):
        raise ValueError('Anonymous query / game / SGF isolation failed')
    if set(audit['question_player_mapping']) != set(q.question_id) or set(audit['question_player_mapping'].values()) != ids:
        raise ValueError('Audited question identity mapping mismatch')
    write_json({'passed':True,'checks':13,'evaluation_count':0,'protocol_sha256':file_digest(directory/'preregistered_protocol.json')},directory/'preflight.json')
    return protocol,selected,checkpoint,c,q,audit


def fixed_scores(model,cs,qs,device,batch,players,questions,opening):
    """Perform frozen inference only, explicitly disabling gradients and training mode."""
    model.eval()
    with torch.no_grad():
        triplet,_,_=score_embeddings(model,cs,qs,device,batch,players,questions)
    return triplet,fuse(opening,triplet,.9,'z-score')


def run(config):
    """Reserve once, infer every method, then read truth exactly once and close forever."""
    started=time.perf_counter()
    lock.require_unconsumed()
    with dev2_only():
        protocol,selected,checkpoint,c,q,audit=preflight(config)
        directory=lock.directory()
        with lock.claim_once():
            with lock.stage('INFERENCE'):
                runtime=copy.deepcopy(selected)
                runtime['paths']=config['paths']
                cs,cc=cached_store(runtime,'val_candidate',c,'player_id','val_candidates_csv','val_candidate_manifest')
                qs,qc=cached_store(runtime,'val_query',q,'question_id','val_queries_csv','val_query_manifest')
                opening,players,questions,errors,missing=opening_scores(c,q,'player',5,True)
                seed_everything(42,True,selected['training']['num_threads'])
                device=choose_device(config['inference']['device'])
                model=encoder(selected,device)
                model.load_state_dict(checkpoint['model_state_dict'])
                triplet,fusion=fixed_scores(model,cs,qs,device,64,players,questions,opening)
                rng=np.random.default_rng(42)
                random=np.empty((len(questions),len(players)))
                for row in random:
                    row[rng.permutation(len(players))]=np.arange(len(players),0,-1)
                matrices=dict(zip(NAMES,[random,opening,triplet,fusion]))
                for name,score in matrices.items():
                    write_csv(rankings(score,players,questions),directory/(name+'_predictions.csv'))
                # Friendly names requested by the protocol, all aliases of the same inference.
                for name,score in [('random',random),('opening',opening),('triplet',triplet),('fusion',fusion)]:
                    write_csv(rankings(score,players,questions),directory/(name+'_predictions.csv'))
                    if name!='random':
                        np.savez_compressed(directory/(name+'_scores.npz'),scores=score,players=np.array(players),questions=np.array(questions))
                write_json({'all_predictions_saved':True,'questions':questions,'players':players},directory/'inference_complete.json')
            with lock.stage('SCORING'):
                truth=read_csv(config['paths']['val_ground_truth_csv'],['question_id','player_id'])
                if dict(zip(truth.question_id,truth.player_id))!=audit['question_player_mapping'] or len(truth)!=100:
                    raise ValueError('Ground truth not identical to preregistered split')
                values,details={},{}
                for name,score in matrices.items():
                    values[name],details[name]=metrics(score,players,questions,truth)
                    write_csv(details[name],directory/(name+'_question_scores.csv'))
            bootstrap,samples=paired_bootstrap(details[NAMES[1]],details[NAMES[3]],1000,42)
            write_csv(pd.DataFrame(bootstrap),directory/'bootstrap_ci.csv')
            write_csv(samples,directory/'bootstrap_samples.csv')
            write_csv(pd.DataFrame([{'method':name,**v} for name,v in values.items()]),directory/'final_comparison.csv')
            a=details[NAMES[1]].set_index('question_id').top_1.astype(bool)
            b=details[NAMES[2]].set_index('question_id').top_1.astype(bool).reindex(a.index)
            f=details[NAMES[3]].set_index('question_id').top_1.astype(bool).reindex(a.index)
            overlap={'opening_only_correct':int((a&~b).sum()),'triplet_only_correct':int((~a&b).sum()),
                     'both_correct':int((a&b).sum()),'both_wrong':int((~a&~b).sum()),
                     'fusion_correct_when_opening_wrong':int((~a&f).sum())}
            write_json(overlap,directory/'error_overlap.json')
            stability=read_json('outputs/phase29/summary.json')
            generalization=[{'method':name,'dev2':ref['dev2_score'],
                             'stability':ref['stability_score'],'final_test3':values[method]['competition_score']}
                            for name,method,ref in zip(['opening','triplet','fusion'],NAMES[1:],stability['comparison'])]
            write_csv(pd.DataFrame(generalization),directory/'generalization_table.csv')
            if file_digest(CHECKPOINT)!=protocol['bindings'][CHECKPOINT] or file_digest(config['selected_config'])!=protocol['bindings'][config['selected_config']]:
                raise ValueError('Frozen artifact changed during inference')
            receipt={'status':'CLOSED TEST','evaluation_count':1,'timestamp':datetime.now(timezone.utc).isoformat(),
                     'dataset_sha256':protocol['bindings'][config['paths']['training_csv']],
                     'split_sha256':protocol['bindings']['outputs/phase210/splits/split_audit.json'],
                     'protocol_sha256':file_digest(directory/'preregistered_protocol.json'),
                     'config_sha256':protocol['bindings'][config['selected_config']],
                     'run_config_sha256':protocol['bindings']['configs/phase210.yaml'],
                     'checkpoint_sha256':protocol['bindings'][CHECKPOINT],'metrics':values,'feature_coverage':[cc,qc],
                     'bootstrap':bootstrap,'error_overlap':overlap,'generalization':generalization,
                     'opening_failed_games':len(errors),'missing_candidate_colors':missing,
                     'elapsed_seconds':time.perf_counter()-started,'split_counts':{k:audit[k] for k in ['players','max_eligible_players','unseen_players','historical_unique_players','candidate_games','query_games']}}
            write_csv(pd.DataFrame(errors,columns=['group_id','game_id','error']),directory/'opening_errors.csv')
            write_json(receipt,directory/'final_test3_receipt.json')
            with (directory/'CLOSED_TEST').open('x',encoding='utf-8') as stream:
                stream.write('Permanent CLOSED TEST; use saved metrics only.\n')
            print('FINAL TEST 3 — ONE-SHOT RESULT',flush=True)
            print(json.dumps({'metrics':values,'bootstrap':bootstrap,'error_overlap':overlap,'elapsed_seconds':receipt['elapsed_seconds']},indent=2),flush=True)
            return receipt


def main():
    """Expose separate preregistration and one-shot commands, with no search controls."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/phase210.yaml'))
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    config=load_config(args.config)
    lock.require_unconsumed()
    if args.prepare:
        registration(config)
        print('PREREGISTERED; all 13 preflight gates passed; evaluation_count=0')
    else:
        run(config)


if __name__=='__main__':
    main()
