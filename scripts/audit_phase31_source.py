"""Read clone/checkpoint metadata and public releases; never deserialize checkpoints."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
COMMIT='b44b70f53de6cdaa7e2f44a590d541492148a9ad'

def main():
    """Audit local user weights separately from tracked-source availability."""
    source=ROOT/'external_refs/minizero_policydetection'
    identity=(ROOT/'outputs/phase31/source_git_identity.txt').read_text().splitlines()
    if identity[0]!=COMMIT:raise SystemExit('Wrong source commit')
    records=[]
    for p in sorted(source.rglob('*')):
        if p.is_file() and '.git' not in p.parts and p.suffix.lower() in ['.pkl','.pt','.pth','.ckpt','.onnx']:
            records.append({'path':str(p.relative_to(source)),'bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'loaded':False})
    request=urllib.request.Request('https://api.github.com/repos/b08202011/minizero_policydetection/releases',headers={'User-Agent':'phase31-audit'})
    with urllib.request.urlopen(request,timeout=45) as response:releases=json.load(response)
    links=[]
    for p in source.rglob('*'):
        if p.is_file() and '.git' not in p.parts and p.suffix.lower() in ['.md','.sh','.py']:
            for number,line in enumerate(p.read_text(encoding='utf-8',errors='replace').splitlines(),1):
                if 'http' in line and any(s in line.lower() for s in ['drive.google','dropbox','huggingface','download','releases']):
                    links.append({'path':str(p.relative_to(source)),'line':number,'text':line.strip()[:600]})
    reference='dan_training_new/model/weight_iter_400000.pkl'
    cfg='dan_training_new/go_19x19_gaz_6bx256_n18-0c403e.cfg'
    known=json.loads((ROOT/'external_refs/policy/audit_manifest.json').read_text(encoding='utf-8'))
    for file in ['minizero/network/py/alphazero_network.py','minizero/network/py/network_unit.py']:
        if hashlib.sha256((source/file).read_bytes()).hexdigest()!=known['sources'][file]['sha256']:
            raise SystemExit('Pinned network source hash mismatch')
    audit={'source_commit':identity[0],'git_tree_hash':identity[1],'clone_path':str(source),
        'checkpoint_files':records,'checkpoint_status':'NOT AVAILABLE' if not records else 'METADATA AUDIT REQUIRED',
        'hardcoded_checkpoint':reference,'hardcoded_checkpoint_exists':(source/reference).is_file(),
        'referenced_cfg':cfg,'referenced_cfg_exists':(source/cfg).is_file(),
        'release_count':len(releases),'release_assets':[a for r in releases for a in r.get('assets',[])],
        'download_link_candidates':links,'verified_pretrained_download_link':None,
        'network_sources_match_pinned_sha256':True,'checkpoint_deserialization_performed':False}
    (ROOT/'outputs/phase31/source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['source_commit','git_tree_hash','checkpoint_status','referenced_cfg_exists','release_count']},indent=2))

if __name__=='__main__':main()
