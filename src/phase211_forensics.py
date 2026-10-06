"""Fixed-checkpoint DEV2 collapse forensics; no training or competition model selection."""
import argparse
from datetime import datetime,timezone
import json
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .embedding_forensics import (LAYERS,TrainGeometry,unit,capture_representations,
    extract_representations,basic_geometry,game_pair_geometry,sample_position_pairs,
    position_pair_geometry,retrieve,color_retrieve,player_separability,spectral_statistics)
from .phase211_guard import dev_only,closed_markers
from .phase26_common import MemoryStore
from .phase28_common import encoder,metrics,rankings,fuse
from .phase28_opening import opening_scores
from .metric_utils import choose_device,file_digest,seed_everything,write_json
from .utils import ROOT,load_config,read_csv,write_csv

CHECKPOINT='outputs/phase28/experiments/Triplet-Hard-100/best.pt'
CHECKPOINT_SHA='9048144a9bc7d422c8c321d607f73fd1dc378054988741e975b56fbb7dab2d4f'


def preflight(config):
    """Verify the unchanged checkpoint, forward code and original DEV2 data/cache provenance."""
    expected=load_config(ROOT/'configs/phase211.yaml')
    if config!=expected or config['checkpoint']!=CHECKPOINT or config['checkpoint_sha256']!=CHECKPOINT_SHA or config['epoch']!=18:
        raise ValueError('Only the fixed Phase 2.11 DEV-only diagnostic protocol is allowed')
    if config['seed']!=42 or config['diagnostics']['remove_pcs']!=[0,1,2,4,8] or config['diagnostics']['whitening_eigenvalue_floor']!=1e-5:
        raise ValueError('Diagnostic seed/PC budget/epsilon cannot be searched')
    if config['fusion']!={'opening_mode':'player','opening_window':5,'color_aware':True,'normalization':'z-score','alpha':.9}:
        raise ValueError('Diagnostic fusion must preserve frozen Opening/alpha')
    closed=closed_markers()
    selected=load_config(ROOT/config['selected_config'])
    if not selected['selection']['frozen'] or selected['selection']['triplet']['best_epoch']!=18 or selected['selection']['triplet']['checkpoint']!=CHECKPOINT:
        raise ValueError('Phase 2.8 fixed selection changed')
    if file_digest(CHECKPOINT)!=CHECKPOINT_SHA or selected['selection']['triplet']['checkpoint_sha256']!=CHECKPOINT_SHA:
        raise ValueError('Checkpoint SHA256 mismatch')
    previous=json.loads((ROOT/'outputs/phase28/protocol_verification.json').read_text(encoding='utf-8'))
    for path in ['src/player_model.py','src/phase28_common.py','src/phase28_triplet.py','configs/phase28_selected.yaml']:
        if file_digest(path)!=previous['code_config_sha256'][path]:
            raise ValueError('Production model/frozen code/config changed')
    for name,digest in selected['selection']['split_sha256'].items():
        if file_digest(ROOT/'outputs/phase28/splits'/name)!=digest:
            raise ValueError('Original DEV2 split SHA256 changed')
    checkpoint=torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=True)
    if checkpoint['epoch']!=18 or checkpoint['model_config']!=selected['model'] or checkpoint['feature_config']!=selected['features']:
        raise ValueError('Checkpoint epoch/model/features mismatch')
    if any(selected['model'][k]!=v for k,v in [('channels',64),('num_blocks',8),('embedding_dim',128)]):
        raise ValueError('Fixed model must be 64 channels / 8 blocks / 128 embedding')
    frames={part:read_csv(selected['paths'][key],required) for part,key,required in [
        ('train','metric_train_csv',['player_id','game_id','color','sgf_content']),
        ('val_candidate','val_candidates_csv',['player_id','game_id','color','sgf_content']),
        ('val_query','val_queries_csv',['question_id','game_id','color','sgf_content']),
        ('truth','val_ground_truth_csv',['question_id','player_id'])]}
    train_ids=set(frames['train'].player_id)
    val_ids=set(frames['val_candidate'].player_id)
    if train_ids&val_ids or not set(checkpoint['training_players'])<=train_ids or len(train_ids)!=200 or len(val_ids)!=100:
        raise ValueError('DEV2 fitting/validation player identity audit failed')
    return selected,checkpoint,frames,closed


def run(config):
    """Infer three unchanged layers once, then explore TRAIN-fit transforms on DEV2 only."""
    started=time.perf_counter()
    with dev_only(),threadpool_limits(limits=config['num_threads']):
        selected,checkpoint,frames,closed=preflight(config)
        directory=ROOT/config['output_dir']
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'summary.json').exists():
            raise ValueError('Forensics already complete; render saved DEV results instead of repeating')
        seed_everything(42,True,config['num_threads'])
        device=choose_device(config['device'])
        model=encoder(selected,device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        source_hashes={part:file_digest(selected['paths'][key]) for part,key in
                       [('train','metric_train_csv'),('val_candidate','val_candidates_csv'),('val_query','val_queries_csv'),('truth','val_ground_truth_csv')]}
        config_sha=file_digest(config['selected_config'])
        stores,position,game,offsets,coverage={},{},{},{},[]
        for part in ['train','val_candidate','val_query']:
            store=MemoryStore(selected,part)
            colors=dict(zip(frames[part].game_id,frames[part].color.str.upper()))
            store.records=[{**record,'color':colors[record['game_id']]} for record in store.records]
            if len(store)!=len(frames[part]):
                raise ValueError('DEV2 feature failures changed the frozen cohort; diagnose errors explicitly')
            stores[part]=store
            coverage.append({'partition':part,'total_games':len(frames[part]),'successful_games':len(store),'failed_games':store.manifest['skipped_games'],
                             'positions':sum(r['num_positions'] for r in store.records)})
            position[part],game[part],offsets[part]=extract_representations(model,store,device,config['batch_size'],directory/'representations',part)
            write_json(store.records,directory/'representations'/(part+'_records.json'))
        # Hooks see the actual production output. Probe both captured and unhooked forward once.
        probe=torch.from_numpy(stores['train'].features(0)[:2].astype(np.float32)).to(device)
        with torch.no_grad():
            before=model(probe).cpu().numpy()
            captured=capture_representations(model,probe)
            after=model(probe).cpu().numpy()
        np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(before,captured['normalized_embedding'].cpu().numpy())
        np.testing.assert_allclose(captured['normalized_embedding'].cpu().numpy(),torch.nn.functional.normalize(captured['raw_embedding'],dim=1).cpu().numpy(),rtol=0,atol=0)
        cr,qr=stores['val_candidate'].records,stores['val_query'].records
        truth=frames['truth']
        mapping=dict(zip(truth.question_id,truth.player_id))
        c_labels=[r['group_id'] for r in cr]
        q_labels=[mapping[r['group_id']] for r in qr]
        players=sorted(frames['val_candidate'].player_id.unique())
        questions=sorted(frames['val_query'].question_id.unique())
        pairs=sample_position_pairs(offsets['val_candidate'],offsets['val_query'],c_labels,q_labels,config['diagnostics']['position_pairs_per_class'],42)
        np.savez_compressed(directory/'position_pair_indices.npz',**pairs)
        geometry=[]
        for layer in LAYERS:
            for level in ['position','game']:
                if level=='position':
                    c,q=position['val_candidate'][layer],position['val_query'][layer]
                    stats,variance=basic_geometry(np.concatenate([c,q]),config['diagnostics']['near_zero_variance'])
                    pair_stats=position_pair_geometry(c,q,pairs)
                else:
                    c,q=game['val_candidate'][layer],game['val_query'][layer]
                    stats,variance=basic_geometry(np.concatenate([c,q]),config['diagnostics']['near_zero_variance'])
                    pair_stats=game_pair_geometry(c,q,c_labels,q_labels)
                geometry.append({'layer':layer,'level':level,'representations':len(c)+len(q),**stats,**pair_stats,
                                 'collapse_warning':stats['near_zero_fraction']>=config['diagnostics']['collapse_fraction']})
                write_csv(pd.DataFrame({'dimension':np.arange(len(variance)),'variance':variance}),directory/(layer+'_'+level+'_variance.csv'))
            print(f'Geometry complete: {layer}',flush=True)
        write_csv(pd.DataFrame(geometry),directory/'layer_geometry.csv')
        train_ids=set(frames['train'].player_id)
        fitting,pca,evals={},[],[]
        for layer in ['raw_embedding','normalized_embedding']:
            fit=TrainGeometry(game['train'][layer],stores['train'].records,'train',train_ids)
            fitting[layer]=fit
            warning=fit.statistics['top1_explained']>=config['diagnostics']['anisotropy_top1'] or fit.statistics['top5_cumulative']>=config['diagnostics']['anisotropy_top5']
            pca.append({'layer':layer,'fit_partition':'DEV2 TRAIN','fit_games':fit.fit_games,'fit_players':len(fit.training_players),
                        **fit.statistics,'total_variance':float(fit.eigenvalues.sum()),'anisotropy_warning':warning,
                        **{f'eigenvalue_{i+1}':float(value) for i,value in enumerate(fit.eigenvalues[:20])}})
            for rank,value in enumerate(fit.eigenvalues[:20],1):
                evals.append({'layer':layer,'rank':rank,'eigenvalue':value,'explained_variance':value/max(fit.eigenvalues.sum(),1e-30)})
            np.savez_compressed(directory/(layer+'_train_statistics.npz'),mean=fit.mean,mean_black=fit.color_means['B'],mean_white=fit.color_means['W'],
                                eigenvalues=fit.eigenvalues,components=fit.components,training_players=np.array(fit.training_players))
        write_csv(pd.DataFrame(pca),directory/'pca_diagnostics.csv')
        write_csv(pd.DataFrame(evals),directory/'pca_top20_eigenvalues.csv')
        # Every center/PCA transform sees TRAIN-only fits, never VAL labels.
        results,score_bank,representation_bank=[],{},{}
        def evaluate_method(name,c,q,color=False,game_normalize=False):
            """Retain the full common DEV2 axes and save each diagnostic prediction matrix."""
            scores=(color_retrieve(c,q,cr,qr,players,questions) if color else retrieve(c,q,cr,qr,players,questions,game_normalize))
            if scores.shape!=(100,100):
                raise ValueError('A diagnostic changed the DEV2 question/candidate axes')
            values,details=metrics(scores,players,questions,truth)
            row={'method':name,**values}
            results.append(row)
            score_bank[name]=scores
            representation_bank[name]=(unit(c) if game_normalize else c,unit(q) if game_normalize else q)
            safe=name.replace(' ','_')
            np.savez_compressed(directory/'scores'/(safe+'.npz'),scores=scores,players=np.array(players),questions=np.array(questions))
            write_csv(rankings(scores,players,questions),directory/'predictions'/(safe+'.csv'))
            write_csv(details,directory/'question_scores'/(safe+'.csv'))
            print(f'{name}: exploratory DEV2 score={values["competition_score"]:.6f}',flush=True)
            return row
        for name in ['scores','predictions','question_scores']:
            (directory/name).mkdir(exist_ok=True)
        c_raw,q_raw=game['val_candidate']['raw_embedding'],game['val_query']['raw_embedding']
        c_norm,q_norm=game['val_candidate']['normalized_embedding'],game['val_query']['normalized_embedding']
        aggregation=[evaluate_method('Current',c_norm,q_norm),evaluate_method('RawMean',c_raw,q_raw),
                     evaluate_method('RawGameNormalize',c_raw,q_raw,game_normalize=True),
                     evaluate_method('Backbone',game['val_candidate']['backbone_feature'],game['val_query']['backbone_feature'])]
        # Numerical reproduction must agree with the frozen checkpoint's saved DEV2 metric.
        if not np.isclose(aggregation[0]['competition_score'],checkpoint['validation_metrics']['competition_score'],rtol=0,atol=1e-12):
            raise ValueError('Current no longer reproduces the checkpoint DEV2 metric')
        write_csv(pd.DataFrame(aggregation),directory/'aggregation_results.csv')
        raw_fit,norm_fit=fitting['raw_embedding'],fitting['normalized_embedding']
        centered_c,centered_q=raw_fit.transform(c_raw),raw_fit.transform(q_raw)
        centering=[aggregation[0],evaluate_method('Centered-Raw',centered_c,centered_q),
                   evaluate_method('Centered-Normalized',norm_fit.transform(c_norm),norm_fit.transform(q_norm))]
        write_csv(pd.DataFrame(centering),directory/'centering_results.csv')
        pc_rows=[]
        for k in config['diagnostics']['remove_pcs']:
            c,q=raw_fit.transform(c_raw,removed_pcs=k),raw_fit.transform(q_raw,removed_pcs=k)
            row=evaluate_method(f'Remove-PC-{k}',c,q)
            pair=game_pair_geometry(c,q,c_labels,q_labels)
            # Effective rank is fitted on the transformed TRAIN covariance, not VAL truth.
            transformed_train=raw_fit.transform(game['train']['raw_embedding'],removed_pcs=k)
            shifted=transformed_train-transformed_train.mean(axis=0)
            eigenvalues=np.linalg.eigvalsh(shifted.T@shifted/len(shifted))[::-1]
            pc_rows.append({'removed_pcs':k,**{key:row[key] for key in ['top1','top3','top5','competition_score']},
                            'same_diff_separation':pair['cosine_separation'],'effective_rank':spectral_statistics(eigenvalues)['effective_rank']})
        write_csv(pd.DataFrame(pc_rows),directory/'remove_pc_results.csv')
        epsilon=config['diagnostics']['whitening_eigenvalue_floor']
        wc,wq=raw_fit.transform(c_raw,whiten=True,epsilon=epsilon),raw_fit.transform(q_raw,whiten=True,epsilon=epsilon)
        whitened=evaluate_method('Whitened',wc,wq)
        whitening=[{**aggregation[1],'comparison':'Raw'}, {**centering[1],'comparison':'Centered'},
                   {**whitened,'comparison':'Whitened'}]
        write_csv(pd.DataFrame(whitening),directory/'whitening_results.csv')
        coloring=[aggregation[0],evaluate_method('Color-aware Triplet',c_norm,q_norm,color=True)]
        write_csv(pd.DataFrame(coloring),directory/'color_triplet_results.csv')
        cg,qg=norm_fit.transform(c_norm),norm_fit.transform(q_norm)
        cb,qb=norm_fit.transform(c_norm,colors=[r['color'] for r in cr]),norm_fit.transform(q_norm,colors=[r['color'] for r in qr])
        color_centering=[coloring[1],evaluate_method('Color-aware + global centering',cg,qg,color=True),
                         evaluate_method('Color-aware + color-specific centering',cb,qb,color=True)]
        write_csv(pd.DataFrame(color_centering),directory/'color_centering_results.csv')
        post_names=[r['method'] for r in centering[1:]]+[f'Remove-PC-{k}' for k in config['diagnostics']['remove_pcs']]+['Whitened']+[r['method'] for r in color_centering]
        best=max((r for r in results if r['method'] in post_names),key=lambda row:(row['competition_score'],row['method']))
        separability=[{'method':name,**player_separability(representation_bank[name][0],cr)} for name in ['Current','RawMean',best['method']]]
        write_csv(pd.DataFrame(separability),directory/'player_separability.csv')
        opening,op_players,op_questions,errors,missing=opening_scores(frames['val_candidate'],frames['val_query'],'player',5,True)
        if players!=op_players or questions!=op_questions:
            raise ValueError('Diagnostic fusion changed DEV2 axes')
        fusion_rows=[]
        for name,method in [('old_triplet_fusion','Current'),('new_triplet_fusion',best['method'])]:
            values,_=metrics(fuse(opening,score_bank[method],.9,'z-score'),players,questions,truth)
            fusion_rows.append({'method':name,'triplet_diagnostic':method,'alpha':.9,'normalization':'z-score',**values})
        write_csv(pd.DataFrame(fusion_rows),directory/'diagnostic_fusion_results.csv')
        warnings=[]
        if any(row['anisotropy_warning'] for row in pca):
            warnings.append('ANISOTROPY WARNING: TRAIN covariance exceeds preregistered top1/top5 concentration threshold')
        if any(row['collapse_warning'] for row in geometry):
            warnings.append('NEAR-ZERO VARIANCE WARNING: absolute threshold is scale-dependent; inspect raw norms and relative scatter')
        floor_count=int((raw_fit.eigenvalues<epsilon).sum())
        if floor_count:
            warnings.append(f'WHITENING FLOOR: {floor_count}/128 eigenvalues below fixed 1e-5; regularization retained unchanged')
        if missing:
            warnings.append(f'Opening has {missing} missing candidate color banks; existing zero-bank rule retained')
        missing_triplet=sum(not any(r['group_id']==p and r['color']==color for r in cr) for p in players for color in ['B','W'])
        if missing_triplet:
            warnings.append(f'Triplet has {missing_triplet} missing candidate color banks; zero similarity, no opposite-color fallback')
        if file_digest(CHECKPOINT)!=CHECKPOINT_SHA or file_digest(config['selected_config'])!=config_sha:
            raise ValueError('Frozen artifacts changed')
        for part,key in [('train','metric_train_csv'),('val_candidate','val_candidates_csv'),('val_query','val_queries_csv'),('truth','val_ground_truth_csv')]:
            if file_digest(selected['paths'][key])!=source_hashes[part]:
                raise ValueError('Historical DEV2 CSV changed during diagnostics')
        summary={'status':'COMPLETE EXPLORATORY DEV-ONLY FORENSICS','timestamp':datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds':time.perf_counter()-started,'checkpoint_sha256':CHECKPOINT_SHA,'epoch':18,'frozen_config_sha256':config_sha,
                 'config':config,'split_sha256':source_hashes,'closed_markers_present':closed,'closed_test_inputs_or_scores_opened':False,
                 'production_forward_exact_probe_passed':True,'training_performed':False,'model_or_checkpoint_modified':False,
                 'fit_scope':'DEV2 TRAIN 200 players / 4000 equal-position game means only; no VAL fitting',
                 'geometry':geometry,'pca':pca,'aggregation':aggregation,'centering':centering,'remove_pc':pc_rows,
                 'whitening':whitening,'color_triplet':coloring,'color_centering':color_centering,'separability':separability,
                 'diagnostic_best_postprocessing':best,'diagnostic_fusion':fusion_rows,'warnings':warnings,'feature_coverage':coverage,
                 'whitening_epsilon':epsilon,'whitening_floored_eigenvalues':floor_count,'whitening_numerically_finite':True,
                 'missing_triplet_color_banks':missing_triplet,'questions':questions,'players':players,
                 'interpretation_policy':'exploratory DEV2 comparison only; no new frozen model or selected YAML'}
        write_json(summary,directory/'summary.json')
        print(json.dumps({'aggregation':aggregation,'centering':centering,'diagnostic_best':best,'warnings':warnings,'elapsed_seconds':summary['elapsed_seconds']},indent=2),flush=True)
        return summary


def main():
    """Expose only the fixed DEV-only diagnostic; never accept historical TEST paths."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/phase211.yaml'))
    args=parser.parse_args()
    with dev_only():
        config=load_config(args.config)
        run(config)


if __name__=='__main__':
    main()
