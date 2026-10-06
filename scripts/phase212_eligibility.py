"""Count unseen DEV3 eligibility using copied identity metadata only."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase212_guard import dev3_only
from src.metric_utils import file_digest,write_json


def dev3_capacity(counts):
    """Reserve VAL identities first; do not count them again toward the TRAIN budget."""
    n20,n40=int((counts>=20).sum()),int((counts>=40).sum())
    return {'eligible_at_least_20_games':n20,'eligible_at_least_40_games':n40,
            'maximum_train_players_after_50_val':max(0,n20-50) if n40>=50 else None,
            'requested_train_players':150,'requested_val_players':50,
            'can_build_requested_dev3':n40>=50 and n20>=200}


def count_eligibility():
    """Exclude every previous cohort, including FINAL TEST 3, without reading closed inputs."""
    with dev3_only():
        previous=ROOT/'outputs/phase210/eligibility.json'
        final3=ROOT/'outputs/phase210/inference_complete.json'
        metadata=json.loads(previous.read_text(encoding='utf-8'))
        final=json.loads(final3.read_text(encoding='utf-8'))
        groups={key:set(values) for key,values in metadata['groups'].items()}
        groups['final_test3']=set(final['players'])
        old=set().union(*(ids for key,ids in groups.items() if key!='final_test3'))
        if len(old)!=615 or len(groups['final_test3'])!=100 or old&groups['final_test3']:
            raise ValueError('Historical identity metadata counts/intersections changed')
        excluded=old|groups['final_test3']
        path=ROOT/'data/training/train_A.csv'
        digest=file_digest(path)
        if digest!=metadata['dataset_sha256']:
            raise ValueError('Official dataset SHA256 changed')
        source=pd.read_csv(path,dtype=str,usecols=['player_id'])
        if not excluded<=set(source.player_id):
            raise ValueError('Historical identities not found in official source')
        counts=source.loc[~source.player_id.isin(excluded)].groupby('player_id').size()
        result={'dataset_sha256':digest,'historical_unique_players':len(excluded),'unseen_players':len(counts),
                **dev3_capacity(counts),
                'groups':{key:sorted(values) for key,values in groups.items()},
                'metadata_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):file_digest(p) for p in [previous,final3]}}
        write_json(result,ROOT/'outputs/phase212/eligibility.json')
        return result


def main():
    """Stop without creating a DEV split or reducing counts if either allocation fails."""
    result=count_eligibility()
    print({key:value for key,value in result.items() if key not in ['groups','metadata_sha256']})
    if not result['can_build_requested_dev3']:
        blocked={'status':'BLOCKED BY DATA ELIGIBILITY — NOT EVALUATED',
                 'eligibility':{key:value for key,value in result.items() if key not in ['groups','metadata_sha256']},
                 'dev3_split_created':False,'training_performed':False,'historical_test_inputs_or_scores_read':False,
                 'budget_reduced':False,'new_checkpoint_created':False,'final_test4_created':False,
                 'experiments':{name:{'status':'NOT RUN','best_epoch':None,'metrics':None,'geometry':None} for name in ['A0','A1','A2','A3']},
                 'intervention_success':'NOT EVALUATED; do not infer regularization failure from missing data'}
        with dev3_only():
            write_json(blocked,ROOT/'outputs/phase212/blocked_summary.json')
        print('INSUFFICIENT ELIGIBLE IDENTITIES: DEV3 not created; no training; no budget reduction')
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
