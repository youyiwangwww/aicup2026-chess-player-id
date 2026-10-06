"""User-authorized 100/50 DEV3 revision, immutable split and TRAIN diagnostic subset."""
import itertools
import json
import numpy as np
import pandas as pd
from .metric_utils import file_digest,write_json
from .utils import ROOT,read_csv,write_csv


def make_dev3(source,excluded,seed=42):
    """Allocate VAL first, then 100 disjoint TRAIN players; retain exact requested budgets."""
    if source.game_id.duplicated().any() or source.sgf_content.duplicated().any():
        raise ValueError('Duplicate source game ID/exact SGF; no silent deduplication')
    available=source[~source.player_id.isin(excluded)]
    counts=available.groupby('player_id').size()
    n20,n40=int((counts>=20).sum()),int((counts>=40).sum())
    if n40<50 or n20<150:
        raise ValueError(f'Insufficient DEV3: maximum >=20={n20}, >=40={n40}; no count reduction')
    rng=np.random.default_rng(seed)
    val=set(rng.permutation(sorted(counts[counts>=40].index))[:50].tolist())
    train=set(rng.permutation(sorted(set(counts[counts>=20].index)-val))[:100].tolist())
    tr,cs,qs,ts=[],[],[],[]
    for player in sorted(train|val):
        games=available[available.player_id==player].sort_values('game_id')
        games=games.iloc[rng.permutation(len(games))]
        if player in train:
            tr.append(games.iloc[:20])
        else:
            question=f'dev3_q_{len(ts):04d}'
            cs.append(games.iloc[:30])
            qs.append(games.iloc[30:40][['game_id','color','sgf_content']].assign(question_id=question))
            ts.append({'question_id':question,'player_id':player})
    frames=[pd.concat(x,ignore_index=True) for x in [tr,cs,qs]]+[pd.DataFrame(ts)]
    audit={'seed':seed,'train_players':100,'train_games':2000,'val_players':50,'candidate_games':1500,'query_games':500,
           'train_player_ids':sorted(train),'val_player_ids':sorted(val),'train_val_player_overlap':len(train&val),
           'historical_player_overlap':len((train|val)&excluded),'game_id_overlap':{},'sgf_content_overlap':{}}
    for i,j in itertools.combinations(range(3),2):
        for field in ['game_id','sgf_content']:
            audit[field+'_overlap'][f'{i}/{j}']=len(set(frames[i][field])&set(frames[j][field]))
    if audit['train_val_player_overlap'] or audit['historical_player_overlap'] or any(audit['game_id_overlap'].values()) or any(audit['sgf_content_overlap'].values()):
        raise ValueError('DEV3 leakage')
    return frames,audit


def prepare(config):
    """Freeze the user revision and common files before any experiment starts."""
    directory=ROOT/config['output_dir']
    eligibility=json.loads((directory/'eligibility.json').read_text(encoding='utf-8'))
    blocked=json.loads((directory/'blocked_summary.json').read_text(encoding='utf-8'))
    if blocked['training_performed'] or blocked['dev3_split_created']:
        raise ValueError('Original protocol must have been blocked before training')
    for path,digest in eligibility['metadata_sha256'].items():
        if file_digest(path)!=digest:
            raise ValueError('Historical identity metadata changed')
    if file_digest(config['paths']['training_csv'])!=eligibility['dataset_sha256']:
        raise ValueError('Official dataset changed')
    names=['train.csv','val_candidates.csv','val_queries.csv','val_ground_truth.csv']
    audit_path=directory/'splits/split_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        for name,digest in audit['file_sha256'].items():
            if file_digest(directory/'splits'/name)!=digest:
                raise ValueError('Frozen DEV3 split changed; no redraw')
        if audit['config_sha256']!=file_digest('configs/phase212.yaml'):
            raise ValueError('Revised configuration changed')
        if file_digest(directory/'revised_protocol.json')!=audit['revised_protocol_sha256'] or file_digest(directory/'train_diagnostic_subset.json')!=audit['diagnostic_subset_sha256']:
            raise ValueError('Frozen revision/subset changed')
        revision=json.loads((directory/'revised_protocol.json').read_text(encoding='utf-8'))
        for name,digest in revision['original_record_sha256'].items():
            if file_digest(directory/name)!=digest:
                raise ValueError('Original blocked record changed')
        return audit
    if (directory/'experiments').exists():
        raise ValueError('Cannot revise split after experiments exist')
    source=read_csv(config['paths']['training_csv'],['player_id','game_id','rank','color','sgf_content'])
    excluded=set().union(*(set(ids) for ids in eligibility['groups'].values()))
    if len(excluded)!=715 or not excluded<=set(source.player_id) or not set(source.color)<={'B','W'}:
        raise ValueError('Historical identity/color provenance invalid')
    frames,audit=make_dev3(source,excluded,config['seed'])
    audit['historical_intersections']={key:len((set(audit['train_player_ids'])|set(audit['val_player_ids']))&set(ids)) for key,ids in eligibility['groups'].items()}
    for name,frame in zip(names,frames):
        write_csv(frame,directory/'splits'/name)
    rng=np.random.default_rng(42)
    selected=sorted(rng.permutation(audit['train_player_ids'])[:20].tolist())
    subset=[]
    for player in selected:
        group=frames[0][frames[0].player_id==player].sort_values('game_id')
        subset.extend(group.iloc[rng.permutation(len(group))[:5]].game_id.tolist())
    subset_path=directory/'train_diagnostic_subset.json'
    write_json({'partition':'TRAIN ONLY','seed':42,'players':selected,'game_ids':sorted(subset),'games_per_player':5},subset_path)
    revision={'original_train_players':150,'original_val_players':50,'original_status':'BLOCKED BEFORE TRAINING',
              'revised_train_players':100,'revised_val_players':50,'revision_reason':'insufficient unseen eligible identities',
              'revision_authorized_by_user':True,'revision_happened_before_any_A0_A1_A2_A3_training':True,
              'selection':'best epoch by DEV3 normal retrieval competition score only; earliest ties',
              'geometry_success':'at least 3/4 primary directions improved vs A0; effective rank alone not sufficient',
              'variance_definition':'population variance, std=sqrt(var+1e-4)',
              'regularizer_scope':'full source-forward pool once; selected hard negatives never appended',
              'config':config,'config_sha256':file_digest('configs/phase212.yaml'),
              'original_record_sha256':{name:file_digest(directory/name) for name in ['eligibility.json','blocked_summary.json']},
              'diagnostic_subset_sha256':file_digest(subset_path)}
    write_json(revision,directory/'revised_protocol.json')
    audit.update(dataset_sha256=eligibility['dataset_sha256'],config_sha256=revision['config_sha256'],
                 revised_protocol_sha256=file_digest(directory/'revised_protocol.json'),
                 diagnostic_subset_sha256=file_digest(subset_path),file_sha256={name:file_digest(directory/'splits'/name) for name in names})
    write_json(audit,audit_path)
    return audit
