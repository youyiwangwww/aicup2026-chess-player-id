"""Audit saved DEV3 training/provenance without any historical TEST access or inference."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase212_guard import dev3_only
from src.metric_utils import file_digest,write_json


def main():
    """Check 80 epochs, loss decomposition, shared initialization/split and normal-only selection."""
    with dev3_only():
        d=ROOT/'outputs/phase212'
        s=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        revision=json.loads((d/'revised_protocol.json').read_text(encoding='utf-8'))
        protocol=json.loads((d/'training_protocol.json').read_text(encoding='utf-8'))
        audit=json.loads((d/'splits/split_audit.json').read_text(encoding='utf-8'))
        for path,digest in protocol['code_sha256'].items():
            assert file_digest(path)==digest,path
        for name,digest in revision['original_record_sha256'].items():
            assert file_digest(d/name)==digest,name
        for name,digest in audit['file_sha256'].items():
            assert file_digest(d/'splits'/name)==digest,name
        assert file_digest(d/'revised_protocol.json')==audit['revised_protocol_sha256']
        assert file_digest(d/'train_diagnostic_subset.json')==audit['diagnostic_subset_sha256']
        assert not audit['historical_player_overlap'] and not audit['train_val_player_overlap']
        assert not any(audit['game_id_overlap'].values()) and not any(audit['sgf_content_overlap'].values())
        assert not any(audit['historical_intersections'].values())
        train=pd.read_csv(d/'splits/train.csv',dtype=str)
        candidates=pd.read_csv(d/'splits/val_candidates.csv',dtype=str)
        queries=pd.read_csv(d/'splits/val_queries.csv',dtype=str)
        assert len(train)==2000 and train.groupby('player_id').size().eq(20).all() and train.player_id.nunique()==100
        assert len(candidates)==1500 and candidates.groupby('player_id').size().eq(30).all() and candidates.player_id.nunique()==50
        assert len(queries)==500 and queries.groupby('question_id').size().eq(10).all() and queries.question_id.nunique()==50
        assert not {'player_id','rank'}&set(queries.columns)
        subset=json.loads((d/'train_diagnostic_subset.json').read_text(encoding='utf-8'))
        assert len(subset['players'])==20 and len(subset['game_ids'])==100
        diagnostic=train[train.game_id.isin(subset['game_ids'])]
        assert diagnostic.groupby('player_id').size().eq(5).all()
        all_hashes=[]
        for index,row in enumerate(s['retrieval']):
            e=d/'experiments'/row['experiment']
            log=pd.read_csv(e/'training_log.csv')
            geom=pd.read_csv(e/'geometry_log.csv')
            assert log.epoch.tolist()==list(range(1,21)) and geom.epoch.tolist()==list(range(1,21))
            assert log.samples_per_epoch.eq(2000).all() and log.source_forward_rows_per_batch.eq(48).all()
            assert log.min_anchor_exposure.eq(20).all() and log.max_anchor_exposure.eq(20).all()
            expected=log.train_triplet_loss+.1*log.train_mean_loss+log.train_variance_loss
            np.testing.assert_allclose(log.train_total_loss,expected,rtol=1e-6,atol=1e-7)
            if index in [0,2]:
                assert log.train_mean_loss.eq(0).all()
            if index in [0,1]:
                assert log.train_variance_loss.eq(0).all()
            assert int(log.loc[log.val_competition_score.idxmax(),'epoch'])==row['best_epoch']
            checkpoint=torch.load(e/'best.pt',map_location='cpu',weights_only=True)
            all_hashes.append(checkpoint['initial_model_sha256'])
            assert checkpoint['shared_split_sha256']==audit['file_sha256']
            assert checkpoint['diagnostic_subset_sha256']==audit['diagnostic_subset_sha256']
            assert checkpoint['epoch']==row['best_epoch']
            assert checkpoint['selection']=='DEV3 normal competition score only'
            assert geom.diagnostic_games.eq(100).all() and geom.diagnostic_positions.eq(1600).all()
            assert geom.fit_partition.eq('TRAIN ONLY').all()
            with np.load(e/'best_scores.npz') as scores:
                assert scores['normal'].shape==(50,50) and scores['color'].shape==(50,50)
                assert scores['players'].tolist()==sorted(candidates.player_id.unique())
                assert scores['questions'].tolist()==sorted(queries.question_id.unique())
        assert len(set(all_hashes))==1 and all_hashes[0]==s['initial_model_sha256']
        fused=pd.read_csv(d/'fusion_diagnostic.csv')
        assert fused.alpha.eq(.9).all() and fused.normalization.eq('z-score').all()
        denied=[]
        for path in ['outputs/splits/test_candidates.csv','outputs/phase26/splits/final_test2_ground_truth.csv',
                     'outputs/phase210/fusion_scores.npz','outputs/phase29/splits/ground_truth.csv',
                     'outputs/phase28/experiments/Triplet-Hard-100/best.pt','outputs/phase28/splits/val_ground_truth.csv']:
            try:
                with (ROOT/path).open('rb'):
                    raise AssertionError('Historical input opened')
            except PermissionError:
                denied.append(path)
        report={'status':'VERIFIED PHASE 2.12-R DEV3 RESEARCH','epochs_verified':80,'from_scratch_identical_initialization':True,
                'shared_split_and_fixed_subset':True,'original_blocked_records_unchanged':True,'loss_decomposition_passed':True,
                'best_epoch_normal_score_only':True,'historical_path_refusal_checks':len(denied),'no_test_inference':True}
        write_json(report,d/'protocol_verification.json')
        print(report)


if __name__=='__main__':
    main()
