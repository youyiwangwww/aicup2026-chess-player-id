"""Summarize already-downloaded public-source evidence; no network or model execution."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    """Save pinned evidence and the complete CPU schema separately from ignored source copies."""
    from src.policy_fingerprint import feature_schema
    records={}
    for key in ['official','strength','policy']:
        m=json.loads((ROOT/'external_refs'/key/'audit_manifest.json').read_text(encoding='utf-8'))
        matches=[r['path'] for r in m['tree'] if Path(r['path']).suffix.lower() in ['.pt','.pth','.pkl','.onnx','.bin','.h5']]
        records[key]={'repo':m['repo'],'branch':m['branch'],'commit':m['commit'],
            'tree_truncated':m['tree_truncated'],'tree_entries':len(m['tree']),
            'release_count':len(m['releases']),'checkpoint_extension_matches':matches,
            'source_files':m['sources'],
            'checkpoint_download_link_verified':False,
            'link_review_scope':'downloaded README, configs and training/build/data scripts only; not the whole Internet'}
    data={'audit_date':'2026-10-07','repositories':records,
          'policy_hardcoded_checkpoint':'dan_training_new/model/weight_iter_400000.pkl',
          'policy_hardcoded_checkpoint_in_tree':False,
          'weights_downloaded':False,'external_code_executed':False}
    out=ROOT/'outputs/phase30'
    out.mkdir(parents=True,exist_ok=True)
    for target in [out/'source_audit.json',ROOT/'docs/phase30_source_audit.json']:
        target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'policy_schema_v1.json').write_text(json.dumps({'version':'policy_fingerprint_v1',
        'phase_boundaries':[30,150],'numeric_feature_count':len(feature_schema()),
        'numeric_feature_names':feature_schema()},indent=2),encoding='utf-8')

if __name__=='__main__':main()
