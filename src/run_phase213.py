"""Multi-seed stability on frozen DEV3 caches; seed 42 is reference-only."""
import argparse
import copy
import json
from datetime import datetime,timezone
import time
import sys
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from .anti_collapse import NAMES,objective,is_best
from .phase213_guard import research_only
from .phase213_analysis import GEOMETRY,METRICS,paired_deltas,aggregate,sign_consistency,gradient_values,complementarity
from .phase212_geometry import GameSubset,train_monitor,capture_games,best_geometry
from .phase28_triplet import ExposureTriplets,hard_negative,score_embeddings
from .phase28_common import encoder,metrics,rankings,fuse
from .embedding_forensics import color_retrieve
from .phase26_common import MemoryStore
from .feature_cache import load_store
from .train_triplet import save_checkpoint
from .run_phase212 import model_digest,validate_config
from .metric_utils import choose_device,file_digest,seed_everything,write_json
from .utils import ROOT,load_config,read_csv,write_csv

SEEDS=[42,123,2026,31415]


def validate_settings(c):
    """Reject seed/split/hyperparameter searches and extra CLI configurations."""
    expected={'protocol':'Phase 2.13','status':'DEV RESEARCH ONLY','split_seed':42,
              'training_seeds':SEEDS,'reference_config':'configs/phase212.yaml',
              'output_dir':'outputs/phase213','seed42_reference_only':True,
              'aggregate_std_ddof':1,'no_test_evaluation':True,'no_new_split':True}
    if c!=expected:
        raise ValueError('Fixed Phase 2.13 configuration changed')


def frozen_snapshot():
    """Validate original recorded hashes and bind every reference artifact, including cache shards."""
    directory=ROOT/'outputs/phase212'
    protocol=json.loads((directory/'training_protocol.json').read_text(encoding='utf-8'))
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='COMPLETE PHASE 2.12-R DEV3 RESEARCH':
        raise ValueError('Require completed original experiment')
    expected={**protocol['code_sha256'],**{'outputs/phase212/splits/'+path:digest for path,digest in protocol['split_sha256'].items()}}
    for path,digest in expected.items():
        if file_digest(path)!=digest:
            raise ValueError('Reference code/config/split SHA256 changed: '+path)
    cache=json.loads((directory/'feature_cache_audit.json').read_text(encoding='utf-8'))
    config=load_config(ROOT/'configs/phase212.yaml')
    for part,key in [('train','train_manifest'),('val_candidate','val_candidate_manifest'),('val_query','val_query_manifest')]:
        if file_digest(config['paths'][key])!=cache['manifest_sha256'][part]:
            raise ValueError('Frozen cache manifest changed')
        load_store(config,part)  # Existing-only validation; never invoke preprocessing.
    for row in summary['retrieval']:
        directory_exp=directory/'experiments'/row['experiment']
        state=json.loads((directory_exp/'state.json').read_text(encoding='utf-8'))
        if not state['complete'] or state['last_epoch']!=20 or file_digest(directory_exp/'best.pt')!=state['best_checkpoint_sha256']:
            raise ValueError('Original checkpoint/state changed')
    paths=[p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(paths)},summary


def config_for_seed(seed):
    """Vary training RNG only; cache validation seed and all data paths remain 42/original."""
    if seed not in SEEDS[1:]:
        raise ValueError('Seed 42 is saved-results-only; unsupported training seed')
    config=copy.deepcopy(load_config(ROOT/'configs/phase212.yaml'))
    config['training_seed']=seed
    config['output_dir']=f'outputs/phase213/seed_{seed}'
    return config


def train_one(config,name,train,candidates,queries,truth,subset,audit,protocol):
    """Train exactly 20 epochs; resume only the same run, select best by normal DEV3 score."""
    directory=ROOT/config['output_dir']/'experiments'/name
    directory.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps({'name':name,'config':config,'protocol':protocol},sort_keys=True).encode()).hexdigest()
    seed_everything(config['training_seed'],True,4)
    device=choose_device('cuda')
    model=encoder(config,device)
    initial=model_digest(model)
    players=sorted({r['group_id'] for r in candidates.records})
    questions=sorted({r['group_id'] for r in queries.records})
    dataset=ExposureTriplets(train,20,config['training_seed'])
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
    print(f'seed={config["training_seed"]} {name}: random init={initial[:12]}, start_epoch={start_epoch}, samples=2000/epoch, source forward rows=48/batch',flush=True)
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
                    'seed':config['training_seed'],'signature':signature,'initial_model_sha256':initial,'training_players':dataset.players,
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
        print(f'seed={config["training_seed"]} {name} epoch={epoch:02d} total={row["train_total_loss"]:.6f} triplet={row["train_triplet_loss"]:.6f} mean={row["train_mean_loss"]:.6f} var={row["train_variance_loss"]:.6f} DEV3={values["competition_score"]:.6f} best={improved} mean_norm={diagnostics["mean_direction_norm"]:.6f} seconds={row["elapsed_seconds"]:.1f}',flush=True)
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




def batch_diagnostic(config,checkpoint_path,batch,seed,name,device,batch_sha):
    """Measure TRAIN-only gradients with eval-mode BN and no optimizer/backprop to parameters."""
    model=encoder(config,device)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    captured=[]
    handle=model.projection.register_forward_hook(lambda module,inputs,output:captured.append(output))
    try:
        with torch.enable_grad():
            z=model(batch.to(device))
            values=gradient_values(captured[0],z)
    finally:
        handle.remove()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Gradient diagnostic populated parameter gradients')
    return {'seed':seed,'experiment':name,'partition':'TRAIN ONLY','batch_rows':len(batch),
            'batch_sha256':batch_sha,'checkpoint_sha256':file_digest(checkpoint_path),
            'best_epoch':checkpoint['epoch'],**values}


def export(directory,rows,geometry,color,fusions,overlaps,gradients,opening_metrics,complete=False):
    """Save paired tables incrementally; issue final stability labels only for all four seeds."""
    results=pd.DataFrame(rows)
    geo=pd.DataFrame(geometry)
    paired=paired_deltas(results,geo)
    aggregated=aggregate(results,geo)
    for filename,frame in [('per_seed_results.csv',results),('per_seed_geometry.csv',geo),
                           ('paired_deltas.csv',paired),('aggregate_results.csv',aggregated),
                           ('color_results.csv',pd.DataFrame(color)),('fusion_results.csv',pd.DataFrame(fusions)),
                           ('error_overlap.csv',pd.DataFrame(overlaps))]:
        write_csv(frame,directory/filename)
    if gradients:
        gradient_frame=pd.DataFrame(gradients)
        write_csv(gradient_frame[gradient_frame.experiment==NAMES[1]],directory/'mean_gradient_diagnostic.csv')
        write_csv(gradient_frame,directory/'variance_gradient_diagnostic.csv')
    if not complete:
        return None
    labels=sign_consistency(paired,aggregated)
    write_csv(pd.DataFrame(labels),directory/'sign_consistency.csv')
    return {'status':'COMPLETE Phase 2.13 DEV RESEARCH ONLY','training_seeds':SEEDS,'split_seed':42,
            'retrieval':rows,'geometry':geometry,'paired_deltas':paired.to_dict('records'),
            'aggregate':aggregated.to_dict('records'),'sign_consistency':labels,'gradients':gradients,
            'color_results':color,'fusion_results':fusions,'error_overlap':overlaps,
            'opening_reference':opening_metrics,'std_ddof':1,'no_new_split':True,
            'no_new_preprocessing':True,'seed42_not_retrained':True,'no_final_model':True,
            'final_test4_created':False,'no_test_evaluation':True}


def run(settings,check_only=False):
    """Read frozen reference/cache once, train only three new seeds, then diagnostic aggregation."""
    started=time.perf_counter()
    with research_only(),threadpool_limits(limits=4):
        validate_settings(settings)
        base=load_config(ROOT/settings['reference_config'])
        validate_config(base)
        snapshot,reference=frozen_snapshot()
        directory=ROOT/settings['output_dir']
        directory.mkdir(parents=True,exist_ok=True)
        protocol={'settings':settings,'phase212_snapshot_sha256':snapshot,
                  'new_code_sha256':{path:file_digest(path) for path in
                   ['src/run_phase213.py','src/phase213_guard.py','src/phase213_analysis.py','configs/phase213.yaml']},
                  'original_split_sha256':reference['split']['file_sha256'],
                  'split_seed':42,'cache_seed':42,'training_rng_controls':['model','epoch permutation','position sampling','PyTorch','CUDA'],
                  'no_preprocessing':True,'no_seed42_training':True}
        protocol_path=directory/'protocol.json'
        if protocol_path.exists():
            if json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
                raise ValueError('Immutable experiment protocol mismatch')
        else:
            write_json(protocol,protocol_path)
        if check_only:
            print('Phase 2.13 safety check passed: frozen split/cache/reference hashes unchanged; no training/preprocessing')
            return
        if (directory/'summary.json').exists():
            raise ValueError('Phase 2.13 complete; use saved results, no repeated training')
        audit=reference['split']
        truth=read_csv(base['paths']['val_ground_truth_csv'],['question_id','player_id'])
        stores={part:MemoryStore(base,part) for part in ['train','val_candidate','val_query']}
        for part,key in [('train','metric_train_csv'),('val_candidate','val_candidates_csv'),('val_query','val_queries_csv')]:
            frame=read_csv(base['paths'][key],['game_id','color'])
            colors=dict(zip(frame.game_id,frame.color.str.upper()))
            stores[part].records=[{**r,'color':colors[r['game_id']]} for r in stores[part].records]
        if [len(stores[p]) for p in stores]!=[2000,1500,500]:
            raise ValueError('Frozen game budget changed')
        subset_meta=json.loads((ROOT/'outputs/phase212/train_diagnostic_subset.json').read_text(encoding='utf-8'))
        subset=GameSubset(stores['train'],subset_meta['game_ids'])
        diagnostic_dataset=ExposureTriplets(stores['train'],20,42)
        a,p,n,_=next(iter(DataLoader(diagnostic_dataset,batch_size=16,shuffle=False,num_workers=0)))
        fixed_batch=torch.cat([a,p,n])
        batch_sha=hashlib.sha256(fixed_batch.numpy().tobytes()).hexdigest()
        diagnostic_pool={'source':'frozen DEV3 TRAIN only','sampling_seed':42,'epoch':0,'indices':list(range(16)),
                         'roles':['anchor','positive','negative'],'rows':48,'tensor_sha256':batch_sha,
                         'game_position_indices':[diagnostic_dataset.sample_indices(i) for i in range(16)]}
        batch_path=directory/'gradient_batch.json'
        if batch_path.exists() and json.loads(batch_path.read_text(encoding='utf-8'))!=json.loads(json.dumps(diagnostic_pool)):
            raise ValueError('Fixed gradient diagnostic batch changed')
        if not batch_path.exists():
            write_json(diagnostic_pool,batch_path)
        opening_npz=np.load(ROOT/'outputs/phase212/opening_scores.npz',allow_pickle=False)
        opening=opening_npz['scores']
        players=opening_npz['players'].tolist()
        questions=opening_npz['questions'].tolist()
        opening_metrics,_=metrics(opening,players,questions,truth)
        if opening_metrics!=reference['opening_reference']:
            raise ValueError('Saved Opening reference differs')
        write_json({'source':'outputs/phase212/opening_scores.npz','sha256':file_digest(ROOT/'outputs/phase212/opening_scores.npz'),
                    'metrics':opening_metrics,'recomputed':False},directory/'opening_reference.json')
        rows,geometry,color,fusions,overlaps,gradients=[],[],[],[],[],[]
        device=choose_device('cuda')
        for seed in SEEDS:
            seed_dir=directory/f'seed_{seed}'
            seed_dir.mkdir(exist_ok=True)
            if seed==42:
                write_json({'seed':42,'reference_only':True,'checkpoint_copied':False,'training_performed':False,
                            'source':'outputs/phase212','original_summary_sha256':file_digest(ROOT/'outputs/phase212/summary.json')},
                           seed_dir/'reference.json')
            initials=[]
            for name in NAMES:
                if seed==42:
                    result=next(r for r in reference['retrieval'] if r['experiment']==name)
                    diag=next(r for r in reference['best_geometry'] if r['experiment']==name)
                    cv=next(r for r in reference['color_retrieval'] if r['experiment']==name and r['retrieval']=='Color-aware')
                    values=np.load(ROOT/'outputs/phase212/experiments'/name/'best_scores.npz',allow_pickle=False)
                    scores=values['normal']
                    checkpoint_path=ROOT/'outputs/phase212/experiments'/name/'best.pt'
                    initial=reference['initial_model_sha256']
                else:
                    config=config_for_seed(seed)
                    result,diag,cv,scores,initial=train_one(config,name,stores['train'],stores['val_candidate'],stores['val_query'],truth,subset,audit,protocol)
                    checkpoint_path=seed_dir/'experiments'/name/'best.pt'
                    values=np.load(seed_dir/'experiments'/name/'best_scores.npz',allow_pickle=False)
                if values['players'].tolist()!=players or values['questions'].tolist()!=questions:
                    raise ValueError('Retrieval/Opening question or player axes mismatch')
                initials.append(initial)
                rows.append({'seed':seed,**result})
                geometry.append({'seed':seed,'experiment':name,**{key:diag[key] for key in GEOMETRY}})
                color.extend([{'seed':seed,'experiment':name,'retrieval':'Normal',**{key:result[key] for key in METRICS}},
                              {'seed':seed,'experiment':name,'retrieval':'Color-aware',**{key:cv[key] for key in METRICS}}])
                fusion=fuse(opening,scores,.9,'z-score')
                fm,_=metrics(fusion,players,questions,truth)
                fusions.append({'seed':seed,'experiment':name,'alpha':.9,'normalization':'z-score',**fm})
                overlaps.append({'seed':seed,'experiment':name,**complementarity(opening,scores,fusion,players,questions,truth)})
                gradients.append(batch_diagnostic(base,checkpoint_path,fixed_batch,seed,name,device,batch_sha))
                print(f'Seed {seed} {name}: score={result["competition_score"]:.6f}, fusion={fm["competition_score"]:.6f}, gradient diagnostic saved',flush=True)
            if len(set(initials))!=1:
                raise ValueError('Within-seed model initialization mismatch')
            if seed!=42:
                write_json({'seed':seed,'status':'DEV RESEARCH ONLY','initial_model_sha256':initials[0],
                            'same_initialization_all_objectives':True,'split_seed':42},seed_dir/'seed_audit.json')
            export(directory,rows,geometry,color,fusions,overlaps,gradients,opening_metrics)
        final_snapshot,_=frozen_snapshot()
        if final_snapshot!=snapshot:
            raise ValueError('Phase 2.12 reference was mutated')
        for path,digest in protocol['new_code_sha256'].items():
            if file_digest(path)!=digest:
                raise ValueError('New training protocol code changed during run')
        summary=export(directory,rows,geometry,color,fusions,overlaps,gradients,opening_metrics,True)
        summary.update(elapsed_seconds=time.perf_counter()-started,timestamp=datetime.now(timezone.utc).isoformat(),
                       phase212_all_artifacts_unchanged=True,protocol_sha256=file_digest(protocol_path))
        write_json(summary,directory/'summary.json')
        print(json.dumps({'status':summary['status'],'sign_consistency':summary['sign_consistency'],
                          'elapsed_seconds':summary['elapsed_seconds']},indent=2),flush=True)


def main():
    """Run the fixed user-authorized seeds or perform a read-only preflight."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/phase213.yaml'))
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    with research_only():
        config=load_config(args.config)
        if args.check:
            run(config,True)
            return
        class Tee:
            """Stream and preserve UTF-8 run logs without altering training."""
            def __init__(self,console,stream):
                self.console,self.stream=console,stream
            def write(self,value):
                self.console.write(value)
                self.stream.write(value)
                self.stream.flush()
            def flush(self):
                self.console.flush()
                self.stream.flush()
        directory=ROOT/config['output_dir']
        directory.mkdir(parents=True,exist_ok=True)
        stdout,stderr=sys.stdout,sys.stderr
        with (directory/'run.log').open('a',encoding='utf-8') as stream:
            sys.stdout,sys.stderr=Tee(stdout,stream),Tee(stderr,stream)
            try:
                run(config)
            finally:
                sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':
    main()
