"""Record the environment blocker without installing packages or fabricating passes."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase31b_pipeline import GATE_NAMES, phase31b_only
from src.minizero_policy_backend import PINNED_COMMIT
OUT=ROOT/'outputs/phase31b'

def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def command(args,timeout=60):
    start=time.perf_counter()
    try:
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        return {'command':args,'exit_code':result.returncode,'stdout':result.stdout,
                'stderr':result.stderr,'elapsed_seconds':time.perf_counter()-start}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'command':args,'exit_code':None,'error':str(e),'elapsed_seconds':time.perf_counter()-start}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    native={'os':platform.platform(),'architecture':platform.machine(),'runtime':{}}
    candidates={'docker':Path('C:/Program Files/Docker/Docker/resources/bin/docker.exe'),
                'podman':Path('C:/Program Files/RedHat/Podman/podman.exe')}
    for tool in ('docker','podman'):
        path=shutil.which(tool) or (str(candidates[tool]) if candidates[tool].is_file() else None)
        native['runtime'][tool]={'path':path,'standard_install_path_exists':candidates[tool].exists()}
        if path:
            native['runtime'][tool]['version']=command([path,'--version'])
            native['runtime'][tool]['info']=command([path,'info','--format','json'])
    probe=command(['wsl.exe','-e','bash','--noprofile','--norc',
        '/mnt/c/Users/abbywang/比賽/ai cup/aicup2026-go-player-id/scripts/probe_phase31b_wsl.sh'])
    save('wsl_probe_command.json',probe)
    if probe['exit_code']!=0:
        raise SystemExit('WSL probe failed; do not classify runtime as absent. See wsl_probe_command.json')
    wsl=json.loads(probe['stdout'])
    available=any(x['path'] for scope in (native,wsl) for x in scope['runtime'].values())
    save('container_audit.json',{'status':'RUNTIME_DETECTED' if available else 'CONTAINER_RUNTIME_NOT_AVAILABLE',
        'windows':native,'wsl':wsl,'installation_performed':False,'image':'kds285/minizero:latest'})
    if available:
        raise SystemExit('Runtime detected: proceed with official image CPU smoke; blocked report intentionally refused')
    source=ROOT/'external_refs/minizero_policydetection'
    if wsl['source_commit']['stdout'].strip()!=PINNED_COMMIT or wsl['source_status']['stdout'].strip():
        raise SystemExit('Pinned source mismatch or tracked source changes; STOP')
    missing=['cmake','Torch/LibTorch','Boost','OpenCV','ALE']
    save('manual_environment_audit.json',{'status':'DEPENDENCIES_MISSING','missing':missing,
        'evidence':wsl['dependencies'],'system_installation_performed':False,
        'native_python':platform.python_version(),'native_torch':__import__('torch').__version__,
        'native_cpu_torch_is_not_wsl_libtorch':True})
    blocked={'status':'BLOCKED','reason':'CONTAINER_RUNTIME_NOT_AVAILABLE','executed':False}
    save('container_environment.json',{**blocked,'os':None,'python':None,'pytorch':None,
        'cuda_build_version':None,'cuda_available':None,'cmake':None,'gcc':None,'g++':None,
        'opencv':None,'boost':None,'ale':None,'torch_cmake_path':None,
        'image_digest':None,'image_size_bytes':None,'pull_seconds':None})
    build={**blocked,'command':'scripts/build.sh go release','command_executed':False,
        'source_commit':PINNED_COMMIT,'source_clean':True,'stdout':'','stderr':'',
        'exit_code':None,'elapsed_seconds':None,'missing_dependency':missing,
        'artifacts':[str(p.relative_to(source)) for p in (source/'build/go').glob('minizero_py*')]}
    save('build_audit.json',build)
    (OUT/'build.log').write_text('NOT RUN: no Docker/Podman runtime; WSL build dependencies missing.\nCommand: scripts/build.sh go release\nexit_code: null\nNo source patches or system installations.\n',encoding='utf-8')
    save('binding_contract.json',{**blocked,'module':'build.go.minizero_py',
        'required_attributes':['load_config_file','Env','DataLoader','TestDataLoader'],'import_executed':False})
    cfg=source/'example.cfg';values={}
    for line in cfg.read_text(encoding='utf-8').splitlines():
        line=line.split('#',1)[0].strip()
        if '=' in line:
            key,value=line.split('=',1)
            if key.strip().startswith('env_'):values[key.strip()]=value.strip()
    feature_cfg=OUT/'feature_only.cfg'
    feature_cfg.write_text('# FEATURE PIPELINE ONLY; not checkpoint matching\n'+''.join(f'{k}={v}\n' for k,v in values.items()),encoding='utf-8')
    save('feature_cfg_audit.json',{'role':'FEATURE PIPELINE ONLY','not_checkpoint_matching':True,
        'source_cfg':str(cfg.relative_to(ROOT)),'source_cfg_sha256':sha(cfg),'settings':values,
        'derived_cfg':str(feature_cfg.relative_to(ROOT)),'derived_cfg_sha256':sha(feature_cfg),
        'load_status':'NOT_RUN','feature_contract_source':'minizero/environment/go/go.h: 18 channels'})
    checkpoints=[str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()
        and '.git' not in p.relative_to(source).parts and p.suffix.lower() in ('.pt','.pth','.pkl','.ckpt','.onnx')]
    save('source_audit.json',{'source_commit':PINNED_COMMIT,'tracked_source_clean':True,
        'files_sha256':{str(p.relative_to(source)):sha(p) for p in
            [source/'scripts/start-container.sh',source/'scripts/build.sh',source/'CMakeLists.txt',
             source/'trainpolicy.sh',source/'minizero/learner/train2.py',source/'minizero/environment/go/go.cpp',
             source/'minizero/network/py/alphazero_network.py']},
        'professor_checkpoint_status':'NOT AVAILABLE' if not checkpoints else 'REQUIRES_VERIFICATION',
        'checkpoint_files':checkpoints,
        'matching_cfg_available':(source/'dan_training_new/go_19x19_gaz_6bx256_n18-0c403e.cfg').exists(),
        'bootstrap_architecture_verified':False,
        'architecture_reason':'trainpolicy.sh cfg name indicates 6bx256_n18; matching cfg absent; class accepts parameters but does not verify missing cfg values. No guessing.'})
    save('real_feature_trace.json',{'status':'BLOCKED','real_features_obtained':False,
        'records':[],'plane_semantics_status':'SOURCE_ONLY_NOT_RUNTIME_VERIFIED',
        'source_definition':{'planes_0_15':'last 8 post-action boards, current-turn own/opponent pairs',
            'plane_16':'all ones on black turn','plane_17':'all ones on white turn',
            'reset':'empty history; planes 0..15 zero','pass':'appends unchanged board; flips turn and shifts history'},
        'pre_move_replay':'NOT_RUN','reproducibility':'NOT_RUN'})
    gates=[{'gate':i,'name':name,'status':'BLOCKED' if i<=14 else 'NOT_RUN',
        'reason':'No container runtime or real binding' if i<=14 else 'Real feature pipeline prerequisite not PASS'}
        for i,name in enumerate(GATE_NAMES,1)]
    save('gates.json',{'status':'ENVIRONMENT BLOCKED','gates':gates,
        'real_feature_pipeline':'BLOCKED','professor_checkpoint_status':'NOT AVAILABLE',
        'only_checkpoint_blocker':False,'phase32_engineering_ready':False})
    save('runtime.json',{'status':'ENVIRONMENT BLOCKED','tiny_bootstrap_executed':False,
        'steps':0,'training_seconds':None,'loss':None,'checkpoint_save_load':'NOT_RUN',
        'aggregation_468':'NOT_RUN','hidden_real_features':'NOT_RUN',
        'player_id_evaluation':False,'final_test_accessed':False})
    print('ENVIRONMENT BLOCKED; no training or real-feature claims')

if __name__=='__main__':
    with phase31b_only(): main()
