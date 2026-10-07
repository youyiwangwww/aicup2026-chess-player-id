"""Structural CPU smoke only; real MiniZero features remain BLOCKED without binding."""
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path):
    """Hash source data without altering it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name,data):
    """Write smoke artifacts only; never retrieval scores."""
    (ROOT/'outputs/phase31'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    """Run actual professor network on synthetic tensors tied only to TRAIN move metadata."""
    import numpy as np
    import pandas as pd
    import torch
    from src.aicup_minizero_adapter import adapt_aicup_row
    from src.minizero_policy_backend import RandomStructuralBackend,hidden_shape_smoke
    from src.policy_move_statistics import extract_move_statistics
    from src.phase31_guard import train_smoke_only
    started=time.perf_counter()
    torch.set_num_threads(4)
    out=ROOT/'outputs/phase31';out.mkdir(parents=True,exist_ok=True)
    hardware=json.loads((out/'hardware_probe.json').read_text(encoding='utf-8-sig'))
    cpu=hardware['cpu'];ram=int(hardware['ram']['TotalPhysicalMemory'])
    save('environment.json',{'device_type':'cpu','device_name':cpu['Name'],
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'cuda_available':torch.cuda.is_available(),'python_version':platform.python_version(),
        'os':platform.platform(),'cpu':cpu,'ram':{'bytes':ram,'gib':ram/1024**3},
        'torch_threads':4,'wsl':'Ubuntu available; missing build dependencies'})
    audit=json.loads((out/'source_audit.json').read_text(encoding='utf-8'))
    if audit['checkpoint_files']:raise SystemExit('Checkpoint metadata/matching cfg must be audited before proceeding')
    dependency_log=(out/'wsl_dependency_probe.log').read_text(encoding='utf-8')
    missing=['cmake','Torch/LibTorch','Boost','OpenCV','ALE']
    save('build_audit.json',{'status':'BLOCKED','wsl_available':True,
        'intended_command':'scripts/build.sh go release','command_executed':False,
        'reason':'Dependency preflight failed; no packages installed',
        'missing_dependency':missing,'exit_code':None,'stdout':dependency_log,'stderr':'',
        'binding_import':'BLOCKED; no build/go/minizero_py artifact'})
    with train_smoke_only():
        train=ROOT/'outputs/phase212/splits/train.csv'
        frame=pd.read_csv(train,dtype={'player_id':str,'game_id':str})
        rng=np.random.default_rng(42)
        selected=sorted(rng.choice(sorted(frame.player_id.unique()),size=min(10,frame.player_id.nunique()),replace=False).tolist())
        pieces=[]
        for player in selected:
            group=frame.loc[frame.player_id==player].sort_values('game_id')
            indices=rng.choice(len(group),size=min(5,len(group)),replace=False)
            pieces.append(group.iloc[sorted(indices)])
        sample=pd.concat(pieces,ignore_index=True)
        save('smoke_manifest.json',{'seed':42,'source_csv':str(train),'source_csv_sha256':sha(train),
            'official_csv_sha256':sha(ROOT/'data/training/train_A.csv'),
            'player_ids_sha256':hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
            'players':len(selected),'games':len(sample),'game_ids':sample.game_id.tolist(),
            'scope':'SUBSET OF EXISTING DEV3 TRAIN; NOT A NEW RESEARCH SPLIT',
            'limits':{'players':10,'games_per_player':5,'total_games':50}})
        # Random structural budget: at most5 games,4targetmoves. No real-feature fallback.
        records=[adapt_aicup_row(row) for row in sample.iloc[:5].to_dict('records')]
        backend=RandomStructuralBackend(ROOT/'external_refs/minizero_policydetection')
        generator=torch.Generator().manual_seed(42)
        features=torch.rand((1,18,19,19),generator=generator)
        output=backend.forward(features)
        if output['policy_logit'].shape!=(1,362) or output['policy'].shape!=(1,362) or not all(torch.isfinite(v).all() for v in output.values()) or not torch.allclose(output['policy'].sum(1),torch.ones(1),atol=1e-6):
            raise RuntimeError('Network contract failed')
        save('network_contract.json',{**backend.metadata(),'input_shape':[1,18,19,19],
            'logit_shape':list(output['policy_logit'].shape),'policy_shape':list(output['policy'].shape),
            'value_shape':list(output['value'].shape),'finite':True,'softmax_sum':output['policy'].sum(1).tolist(),
            'feature_source':'SYNTHETIC RANDOM TENSOR; NOT MINIZERO FEATURES'})
        save('hidden_contract.json',{**hidden_shape_smoke(backend,features),**backend.metadata()})
        diagnostics=[];forward_seconds=0.
        for record in records:
            for move in record.target_moves[:4]:
                tensor=torch.rand((1,18,19,19),generator=generator)
                t=time.perf_counter();logits=backend.extract_logits(tensor)[0].numpy();forward_seconds+=time.perf_counter()-t
                stats=extract_move_statistics(logits,move.action_index,{'game_id':record.game_id,
                    'player_id':record.player_id,'color':move.color,'total_move_number':move.total_move_number,
                    'target_player_move_number':move.target_move_number,'actual_move_coordinate':move.coordinate})
                diagnostics.append({**asdict(stats),'random_weights':True,'meaningful_policy_statistics':False,
                    'feature_source':'SYNTHETIC; NO PRE-MOVE BOARD FEATURES',
                    'label':'STRUCTURAL SMOKE ONLY / NOT POLICY MODEL / NOT RESEARCH RESULT'})
                print(f'structural positions={len(diagnostics)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        save('random_structural_move_diagnostics.json',{'random_weights':True,'meaningful_policy_statistics':False,
            'aggregation_executed':False,'records':diagnostics})
        save('feature_trace.json',{'feature_gate':'BLOCKED','real_features_obtained':False,
            'reason':'C++ dependencies missing; no invented exact18-plane extractor',
            'adapter_move_trace':[{'game_id':r.game_id,'target_color':r.target_color,
                'moves':[asdict(m) for m in r.moves[:8]]} for r in records]})
        names=['build','binding import','SGF read','AI CUP adapter','target moves','network forward',
            'logits Bx362','actual action index','probability','rank','Top1/3/5','entropy','pass','B/W pre-move feature']
        gates=[]
        for i,name in enumerate(names,1):
            status='BLOCKED' if i in [1,2,14] else 'PASS' if i in [3,4,5,8] else 'STRUCTURAL_ONLY'
            artifact='build_audit.json' if i<=2 else 'feature_trace.json' if i in [3,4,5,8,14] else 'network_contract.json' if i in [6,7] else 'tests.log'
            gates.append({'gate':i,'name':name,'status':status,
                'evidence':'Native binding not available' if status=='BLOCKED' else 'CPU adapter/unit contract' if status=='PASS' else 'Random professor network / synthetic unit fixture only',
                'error':'Missing C++ dependencies' if status=='BLOCKED' else None,
                'artifact_path':'outputs/phase31/'+artifact})
        save('gates.json',{'gates':gates,'status':'STRUCTURAL NETWORK PASS / BINDING BLOCKED'})
        save('runtime.json',{'device_type':'cpu','positions':len(diagnostics),'games':len(records),
            'forward_seconds':forward_seconds,'positions_per_second':len(diagnostics)/forward_seconds,
            'batch_size':1,'batch_selection':'Conservative structural batch1;16GB RAM',
            'total_seconds':time.perf_counter()-started,'timing_scope':'Synthetic-tensor random network forward only; NOT SGF feature throughput',
            'tiny_bootstrap_executed':False,'tiny_bootstrap_reason':'Real MiniZero feature pipeline BLOCKED; referenced cfg absent',
            'tiny_bootstrap_loss':None,'random_weights':True,'meaningful_policy_statistics':False,
            'status':'STRUCTURAL NETWORK PASS / BINDING BLOCKED','readiness':'NOT READY FOR PHASE 3.2'})

if __name__=='__main__':main()
