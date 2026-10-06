"""Four from-scratch DEV3 anti-collapse experiments, with immutable common protocol."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import time
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
import yaml

from .anti_collapse import NAMES,objective,is_best,classify
from .phase212_guard import dev3_only
from .phase212_split import prepare
from .phase212_geometry import GameSubset,train_monitor,capture_games,best_geometry
from .phase28_triplet import ExposureTriplets,hard_negative,score_embeddings
from .phase28_common import encoder,metrics,rankings,fuse
from .phase28_opening import opening_scores
from .embedding_forensics import color_retrieve
from .run_phase28 import cached_store
from .train_triplet import save_checkpoint
from .metric_utils import choose_device,file_digest,seed_everything,write_json
from .utils import ROOT,load_config,read_csv,write_csv


def validate_config(c):
    """Reject any revised-budget, objective, architecture or sampling hyperparameter search."""
    expected={'protocol':'Phase 2.12-R','seed':42,'output_dir':'outputs/phase212',
              'split':{'train_players':100,'train_games':20,'val_players':50,'candidate_games':30,'query_games':10},
              'features':{'board_size':19,'history_length':8,'max_positions_per_game':16},
              'model':{'in_channels':17,'channels':64,'num_blocks':8,'embedding_dim':128,'pool_size':3},
              'training':{'device':'cuda','epochs':20,'batch_size':16,'learning_rate':.001,'weight_decay':.0001,'margin':.2,
                          'triplets_per_player_per_epoch':20,'num_threads':4,'deterministic':True},
              'regularizers':{'lambda_mean':.1,'lambda_var':1.,'gamma':'inverse_sqrt_128','variance_epsilon':.0001,'variance_correction':0},
              'diagnostics':{'players':20,'games_per_player':5,'variance_threshold':1e-8},
              'fusion':{'opening_mode':'player','opening_window':5,'color_aware':True,'alpha':.9,'normalization':'z-score'},
              'experiments':NAMES}
    for key,value in expected.items():
        if c[key]!=value:
            raise ValueError(f'Fixed Phase 2.12-R protocol changed: {key}')
    if c!=load_config(ROOT/'configs/phase212.yaml'):
        raise ValueError('Custom paths/configuration are forbidden')


def model_digest(model):
    """Bind identical from-scratch initialization and model/buffer state without checkpoint reuse."""
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_one(config,name,train,candidates,queries,truth,subset,audit,protocol):
    """Train exactly 20 epochs; resume only the same run, select best by normal DEV3 score."""
    directory=ROOT/config['output_dir']/'experiments'/name
    directory.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps({'name':name,'config':config,'protocol':protocol},sort_keys=True).encode()).hexdigest()
    seed_everything(42,True,4)
    device=choose_device('cuda')
    model=encoder(config,device)
    initial=model_digest(model)
    players=sorted({r['group_id'] for r in candidates.records})
    questions=sorted({r['group_id'] for r in queries.records})
    dataset=ExposureTriplets(train,20,42)
    if len(dataset)!=2000 or len(dataset.players)!=100:
        raise ValueError('TRAIN sampling budget changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    state_path=directory/'state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    logs,geometry=[],[]
    best=-1.
    best_epoch=0
    start_epoch=1
    if state:
        if state['signature']!=signature or state['initial_model_sha256']!=initial:
            raise ValueError('Resume signature/initialization mismatch; cannot restart or select another run')
        if file_digest(directory/'best.pt')!=state['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint changed')
        if not state['complete']:
            checkpoint=torch.load(directory/'last.pt',map_location='cpu',weights_only=True)
            if checkpoint['signature']!=signature:
                raise ValueError('Last checkpoint signature mismatch')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            torch.set_rng_state(checkpoint['cpu_rng'])
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
            start_epoch=checkpoint['epoch']+1
        else:
            start_epoch=21
        logs=pd.read_csv(directory/'training_log.csv').to_dict('records')
        geometry=pd.read_csv(directory/'geometry_log.csv').to_dict('records')
        best=state['best_score']
        best_epoch=state['best_epoch']
    print(f'{name}: random init={initial[:12]}, start_epoch={start_epoch}, samples=2000/epoch, source forward rows=48/batch',flush=True)
    for epoch in range(start_epoch,21):
        epoch_start=time.perf_counter()
        dataset.set_epoch(epoch-1)
        model.train()
        sums={key:0. for key in ['total','triplet','mean','variance','batch_mean_norm','std_mean','std_min']}
        seen=0
        exposure=np.zeros(100,dtype=np.int64)
        for a,p,n,labels in DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0):
            batch=len(a)
            pool=model(torch.cat([a,p,n]).to(device))
            aa,pp,_=pool.split(batch)
            labels=labels.to(device)
            nn,selected=hard_negative(aa,pool,labels[:,0],labels.T.reshape(-1))
            parts=objective(aa,pp,nn,pool,name)
            if not torch.isfinite(parts['total']):
                raise ValueError('Nonfinite training loss; no hyperparameter adjustment')
            torch.testing.assert_close(parts['total'],parts['triplet']+.1*parts['mean']+parts['variance'],rtol=1e-6,atol=1e-7)
            optimizer.zero_grad(set_to_none=True)
            parts['total'].backward()
            optimizer.step()
            for key in ['total','triplet','mean','variance']:
                sums[key]+=parts[key].detach().item()*batch
            actual_std=pool.detach().std(dim=0,correction=0)
            sums['batch_mean_norm']+=pool.detach().mean(dim=0).norm().item()*batch
            sums['std_mean']+=actual_std.mean().item()*batch
            sums['std_min']+=actual_std.min().item()*batch
            seen+=batch
            for label in labels[:,0].cpu().tolist():
                exposure[label]+=1
        if seen!=2000 or not np.all(exposure==20):
            raise ValueError('Actual anchor exposure differs from fixed 20/player')
        scores,_,_=score_embeddings(model,candidates,queries,device,64,players,questions)
        values,_=metrics(scores,players,questions,truth)
        improved=is_best(values['competition_score'],best)
        if improved:
            best=values['competition_score']
            best_epoch=epoch
        row={'epoch':epoch,**{'train_'+key+'_loss':sums[key]/seen for key in ['total','triplet','mean','variance']},
             'val_top1':values['top1'],'val_top3':values['top3'],'val_top5':values['top5'],
             'val_competition_score':values['competition_score'],'is_best':improved,
             'embedding_batch_mean_norm':sums['batch_mean_norm']/seen,'dimension_std_mean':sums['std_mean']/seen,
             'dimension_std_min':sums['std_min']/seen,'samples_per_epoch':seen,'source_forward_rows_per_batch':48,
             'min_anchor_exposure':int(exposure.min()),'max_anchor_exposure':int(exposure.max()),
             'elapsed_seconds':time.perf_counter()-epoch_start}
        diagnostics={'epoch':epoch,**train_monitor(model,subset,device,64,1e-8)}
        row['elapsed_seconds']=time.perf_counter()-epoch_start
        logs.append(row)
        geometry.append(diagnostics)
        checkpoint={'epoch':epoch,'model_state_dict':model.state_dict(),'model_config':config['model'],'feature_config':config['features'],
                    'seed':42,'signature':signature,'initial_model_sha256':initial,'training_players':dataset.players,
                    'validation_metrics':values,'selection':'DEV3 normal competition score only',
                    'shared_split_sha256':audit['file_sha256'],'diagnostic_subset_sha256':audit['diagnostic_subset_sha256']}
        if improved:
            save_checkpoint(checkpoint,directory/'best.pt')
        last={**checkpoint,'optimizer_state_dict':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),
              'cuda_rng':torch.cuda.get_rng_state_all()}
        save_checkpoint(last,directory/'last.pt')
        write_csv(pd.DataFrame(logs),directory/'training_log.csv')
        write_csv(pd.DataFrame(geometry),directory/'geometry_log.csv')
        state={'signature':signature,'initial_model_sha256':initial,'last_epoch':epoch,'best_score':best,
               'best_epoch':best_epoch,
               'best_checkpoint_sha256':file_digest(directory/'best.pt'),'complete':epoch==20}
        write_json(state,state_path)
        print(f'{name} epoch={epoch:02d} total={row["train_total_loss"]:.6f} triplet={row["train_triplet_loss"]:.6f} mean={row["train_mean_loss"]:.6f} var={row["train_variance_loss"]:.6f} DEV3={values["competition_score"]:.6f} best={improved} mean_norm={diagnostics["mean_direction_norm"]:.6f} seconds={row["elapsed_seconds"]:.1f}',flush=True)
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    scores,cg,qg=score_embeddings(model,candidates,queries,device,64,players,questions)
    values,_=metrics(scores,players,questions,truth)
    if values!=checkpoint['validation_metrics']:
        raise ValueError('Best checkpoint metric reproduction failed')
    cp,cgames=capture_games(model,candidates,device,64)
    qp,qgames=capture_games(model,queries,device,64)
    np.testing.assert_array_equal(cg,cgames['normalized_embedding'])
    np.testing.assert_array_equal(qg,qgames['normalized_embedding'])
    diagnostic,variance=best_geometry(cgames['raw_embedding'],qgames['raw_embedding'],cg,qg,candidates.records,queries.records,truth,1e-8)
    color=color_retrieve(cg,qg,candidates.records,queries.records,players,questions)
    color_values,_=metrics(color,players,questions,truth)
    write_csv(rankings(scores,players,questions),directory/'normal_predictions.csv')
    write_csv(rankings(color,players,questions),directory/'color_predictions.csv')
    write_csv(pd.DataFrame({'dimension':np.arange(128),'variance':variance}),directory/'best_dimension_variance.csv')
    np.savez_compressed(directory/'best_scores.npz',normal=scores,color=color,players=np.array(players),questions=np.array(questions))
    result={'experiment':name,'best_epoch':checkpoint['epoch'],**values}
    write_json({'retrieval':result,'geometry':diagnostic,'color_metrics':color_values,'best_checkpoint_sha256':file_digest(directory/'best.pt')},directory/'result.json')
    return result,{'experiment':name,**diagnostic},color_values,scores,initial


def run(config,prepare_only=False):
    """Prepare once, train four fixed objectives, then report secondary diagnostics only."""
    started=time.perf_counter()
    with dev3_only(),threadpool_limits(limits=4):
        validate_config(config)
        audit=prepare(config)
        directory=ROOT/config['output_dir']
        if prepare_only:
            print(json.dumps({k:audit[k] for k in ['train_players','train_games','val_players','candidate_games','query_games','train_val_player_overlap','historical_player_overlap']},indent=2))
            return
        if (directory/'summary.json').exists():
            raise ValueError('Phase 2.12-R complete; use saved results, no repeated training')
        code_paths=['src/player_model.py','src/phase28_common.py','src/phase28_triplet.py','src/anti_collapse.py','src/run_phase212.py','src/phase212_geometry.py','src/phase212_split.py','src/phase212_guard.py','configs/phase212.yaml']
        protocol={'config':config,'split_sha256':audit['file_sha256'],'revision_sha256':audit['revised_protocol_sha256'],
                  'diagnostic_subset_sha256':audit['diagnostic_subset_sha256'],
                  'code_sha256':{path:file_digest(path) for path in code_paths},'from_scratch':True,'no_historical_checkpoint':True}
        run_protocol=directory/'training_protocol.json'
        if run_protocol.exists():
            if json.loads(run_protocol.read_text(encoding='utf-8'))!=protocol:
                raise ValueError('Training protocol/source code changed; refusing resume')
        else:
            write_json(protocol,run_protocol)
        frames={part:read_csv(config['paths'][key],required) for part,key,required in [
            ('train','metric_train_csv',['player_id','game_id','color','sgf_content']),
            ('val_candidate','val_candidates_csv',['player_id','game_id','color','sgf_content']),
            ('val_query','val_queries_csv',['question_id','game_id','color','sgf_content']),
            ('truth','val_ground_truth_csv',['question_id','player_id'])]}
        stores,coverage={},[]
        for part,column,source,manifest in [('train','player_id','metric_train_csv','train_manifest'),
                                           ('val_candidate','player_id','val_candidates_csv','val_candidate_manifest'),
                                           ('val_query','question_id','val_queries_csv','val_query_manifest')]:
            stores[part],stats=cached_store(config,part,frames[part],column,source,manifest)
            if len(stores[part])!=len(frames[part]):
                raise ValueError('Feature failures changed game budget; no silent reduction')
            colors=dict(zip(frames[part].game_id,frames[part].color.str.upper()))
            stores[part].records=[{**r,'color':colors[r['game_id']]} for r in stores[part].records]
            coverage.append(stats)
        write_csv(pd.DataFrame(coverage),directory/'feature_coverage.csv')
        write_json({'shared_by':NAMES,'manifest_sha256':{part:file_digest(config['paths'][key]) for part,key in
                   [('train','train_manifest'),('val_candidate','val_candidate_manifest'),('val_query','val_query_manifest')]},
                   'features':config['features'],'seed':42,'single_shared_in_memory_store_per_partition':True},directory/'feature_cache_audit.json')
        subset_metadata=json.loads((directory/'train_diagnostic_subset.json').read_text(encoding='utf-8'))
        subset=GameSubset(stores['train'],subset_metadata['game_ids'])
        opening,players,questions,errors,missing=opening_scores(frames['val_candidate'],frames['val_query'],'player',5,True)
        opening_metrics,_=metrics(opening,players,questions,frames['truth'])
        write_csv(pd.DataFrame([{'method':'Color-aware-player-5',**opening_metrics}]),directory/'opening_reference.csv')
        np.savez_compressed(directory/'opening_scores.npz',scores=opening,players=np.array(players),questions=np.array(questions))
        results,geometry,colors,fusions,initials=[],[],[],[],[]
        for name in NAMES:
            result,diag,color_values,scores,initial=train_one(config,name,stores['train'],stores['val_candidate'],stores['val_query'],frames['truth'],subset,audit,protocol)
            results.append(result)
            geometry.append(diag)
            initials.append(initial)
            colors.extend([{'experiment':name,'retrieval':'Normal',**{k:result[k] for k in ['top1','top3','top5','competition_score']}},
                           {'experiment':name,'retrieval':'Color-aware',**color_values}])
            fm,_=metrics(fuse(opening,scores,.9,'z-score'),players,questions,frames['truth'])
            fusions.append({'experiment':name,'alpha':.9,'normalization':'z-score',**fm})
            write_csv(pd.DataFrame(results),directory/'retrieval_results.csv')
            write_csv(pd.DataFrame(geometry),directory/'best_geometry.csv')
            write_csv(pd.DataFrame(colors),directory/'color_retrieval_results.csv')
            write_csv(pd.DataFrame(fusions),directory/'fusion_diagnostic.csv')
        if len(set(initials))!=1:
            raise ValueError('Four experiments did not share identical initialization')
        matrix=[]
        for result,diag in zip(results,geometry):
            row={**result,**{key:diag[key] for key in ['mean_direction_norm','raw_pc1_explained','normalized_effective_rank','cosine_separation','between_within_ratio']},
                 'retrieval_delta_vs_A0':result['competition_score']-results[0]['competition_score']}
            matrix.append(row)
        for row in matrix:
            row.update(classify(row,matrix[0]))
        matrix[0]['classification']='A0 REFERENCE'
        write_csv(pd.DataFrame(matrix),directory/'summary_matrix.csv')
        promising=[row for row in matrix[1:] if row['classification']=='PROMISING REPRESENTATION INTERVENTION']
        conclusion='PROMISING REPRESENTATION INTERVENTION' if promising else 'SIMPLE ANTI-COLLAPSE REGULARIZATION INSUFFICIENT'
        candidate=max(results,key=lambda row:row['competition_score'])
        dev_candidate={'status':'DEV3 RESEARCH CANDIDATE ONLY','not_final_model':True,'no_test_evaluation':True,
                       'criterion':'highest DEV3 normal retrieval competition score; geometry reported separately',
                       'experiment':candidate['experiment'],'best_epoch':candidate['best_epoch'],
                       'checkpoint':str((directory/'experiments'/candidate['experiment']/'best.pt').relative_to(ROOT)).replace('\\','/'),
                       'metrics':candidate,'geometry_classification':next(row['classification'] for row in matrix if row['experiment']==candidate['experiment'])}
        with (ROOT/'configs/phase212_best_dev.yaml').open('w',encoding='utf-8') as stream:
            yaml.safe_dump(dev_candidate,stream,sort_keys=False,allow_unicode=True)
        for path,digest in protocol['code_sha256'].items():
            if file_digest(path)!=digest:
                raise ValueError('Training code/config changed during experiment')
        prepare(config)
        summary={'status':'COMPLETE PHASE 2.12-R DEV3 RESEARCH','timestamp':datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds':time.perf_counter()-started,'split':audit,'initial_model_sha256':initials[0],
                 'feature_coverage':coverage,'retrieval':results,'best_geometry':geometry,'color_retrieval':colors,
                 'opening_reference':opening_metrics,'fusion_diagnostic':fusions,'summary_matrix':matrix,'conclusion':conclusion,
                 'dev_candidate':dev_candidate,'promising_experiments':[row['experiment'] for row in promising],
                 'opening_failed_games':len(errors),'opening_missing_color_banks':missing,
                 'historical_test_or_dev2_or_stability_used_for_selection':False,'final_test4_created':False,
                 'original_blocked_record_preserved':True,'variance_definition':'population variance + epsilon1e-4',
                 'regularizer_forward_rows':48,'hard_negative_reuse_does_not_duplicate_regularizer':True}
        write_json(summary,directory/'summary.json')
        print(json.dumps({'retrieval':results,'conclusion':conclusion,'elapsed_seconds':summary['elapsed_seconds']},indent=2),flush=True)
        return summary


def main():
    """Start only the user-authorized revised fixed protocol; no search or TEST controls."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/phase212.yaml'))
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    with dev3_only():
        config=load_config(args.config)
        if args.prepare:
            run(config,True)
            return
        class Tee:
            """Keep a UTF-8 training transcript while streaming progress to the console."""
            def __init__(self,console,stream):
                self.console,self.stream=console,stream
            def write(self,value):
                self.console.write(value)
                self.stream.write(value)
                self.stream.flush()
            def flush(self):
                self.console.flush()
                self.stream.flush()
        stdout,stderr=sys.stdout,sys.stderr
        with (ROOT/config['output_dir']/'run.log').open('a',encoding='utf-8') as stream:
            sys.stdout,sys.stderr=Tee(stdout,stream),Tee(stderr,stream)
            try:
                run(config)
            finally:
                sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':
    main()
