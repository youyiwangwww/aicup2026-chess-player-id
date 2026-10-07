#!/usr/bin/env bash
# Read-only audit; deliberately no installs, source patches, or builds.
set -u
cd "$(dirname "$0")/.." || exit 1
python3 - <<'PY'
import glob, importlib.util, json, platform, shutil, subprocess
from pathlib import Path
def command(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=30)
        return {'command':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'command':args,'exit_code':None,'error':str(e)}
r={'os':platform.platform(),'architecture':platform.machine(),'runtime':{},'dependencies':{}}
for tool in ('docker','podman'):
    path=shutil.which(tool)
    r['runtime'][tool]={'path':path}
    if path:
        r['runtime'][tool]['version']=command([path,'--version'])
        r['runtime'][tool]['info']=command([path,'info','--format','json'])
for tool in ('cmake','g++','pkg-config','python3','git'):
    path=shutil.which(tool)
    r['dependencies'][tool]={'path':path,'version':command([path,'--version']) if path else None}
r['dependencies']['torch_spec_available']=importlib.util.find_spec('torch') is not None
if r['dependencies']['torch_spec_available']:
    import torch
    r['dependencies']['torch']={'version':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cmake_prefix':torch.utils.cmake_prefix_path}
r['dependencies']['opencv']=command(['pkg-config','--modversion','opencv4'])
r['dependencies']['boost_header']=Path('/usr/include/boost/version.hpp').exists()
r['dependencies']['ale_files']=glob.glob('/usr/local/lib/libale*')+glob.glob('/usr/include/ale*')
r['dependencies']['cmake_configs']=command(['find','/usr/local','/opt','-maxdepth','6','(','-name','TorchConfig.cmake','-o','-iname','aleconfig.cmake',')'])
source=Path('external_refs/minizero_policydetection')
r['source_commit']=command(['git','-C',str(source),'rev-parse','HEAD'])
r['source_status']=command(['git','-C',str(source),'status','--porcelain','--untracked-files=no'])
print(json.dumps(r,ensure_ascii=False))
PY
