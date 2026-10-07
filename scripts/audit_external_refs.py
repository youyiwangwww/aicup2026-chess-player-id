"""Download pinned, read-only source references; never fetch weights or execute external code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPOS={'official':('AILAB-NDHU/AICup-2026-Tutorial','main'),
       'strength':('rlglab/strength-estimator','iclr2025'),
       'policy':('b08202011/minizero_policydetection','policy')}


def fetch(url):
    """Read public GitHub content without credentials or executing repository scripts."""
    request=urllib.request.Request(url,headers={'User-Agent':'aicup-phase30-static-audit'})
    with urllib.request.urlopen(request,timeout=45) as response:
        return response.read()


def main():
    """Record commit/tree/release metadata, then optionally fetch explicitly named sources."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',choices=REPOS)
    parser.add_argument('--files',nargs='*')
    args=parser.parse_args()
    base=ROOT/'external_refs'
    base.mkdir(exist_ok=True)
    for key in ([args.repo] if args.repo else REPOS):
        repo,branch=REPOS[key]
        target=base/key
        target.mkdir(exist_ok=True)
        manifest_path=target/'audit_manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        else:
            commit=json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/{branch}'))
            sha=commit['sha']
            tree=json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1'))
            releases=json.loads(fetch(f'https://api.github.com/repos/{repo}/releases'))
            manifest={'repo':repo,'branch':branch,'commit':sha,'tree_truncated':tree.get('truncated'),
                      'tree':tree['tree'],'releases':releases,'sources':{}}
        if args.files:
            def download(name):
                path=Path(name)
                if path.is_absolute() or '..' in path.parts or path.suffix.lower() in ['.pt','.pth','.pkl','.onnx']:
                    raise ValueError('Only source references, never checkpoints')
                data=fetch(f'https://raw.githubusercontent.com/{repo}/{manifest["commit"]}/{name}')
                destination=target/path
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(data)
                return name,{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
            with ThreadPoolExecutor(max_workers=4) as pool:
                for name,record in pool.map(download,args.files):
                    manifest['sources'][name]=record
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(key,manifest['commit'],'tree entries',len(manifest['tree']),'releases',len(manifest['releases']))
        if not args.files:
            for row in manifest['tree']:
                if row['type']=='blob' and (row['path'].endswith(('.py','.cfg','.sh','.md','.ipynb')) or 'policy' in row['path'] or 'network' in row['path'] or 'go_action' in row['path']):
                    print(row['path'])


if __name__=='__main__':
    main()
