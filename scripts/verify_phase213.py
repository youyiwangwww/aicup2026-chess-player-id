"""Audit saved multi-seed DEV3 results without inference or checkpoint selection."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.run_phase213 import frozen_snapshot,SEEDS,train_one
from src.run_phase212 import train_one as original_train_one
from src.phase213_guard import research_only
from src.phase213_analysis import paired_deltas,aggregate,sign_consistency
from src.anti_collapse import NAMES
from src.metric_utils import write_json,file_digest


def main():
    """Verify exact reference preservation, RNG-only changes and all saved selection/loss records."""
    with research_only():
        d=ROOT/'outputs/phase213'
        protocol=json.loads((d/'protocol.json').read_text(encoding='utf-8'))
        summary=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        snapshot,reference=frozen_snapshot()
        assert snapshot==protocol['phase212_snapshot_sha256']
        for path,digest in protocol['new_code_sha256'].items():
            assert file_digest(path)==digest
        copied=inspect.getsource(train_one)
        copied=copied.replace("seed_everything(config['training_seed'],True,4)",'seed_everything(42,True,4)')
        copied=copied.replace("ExposureTriplets(train,20,config['training_seed'])",'ExposureTriplets(train,20,42)')
        copied=copied.replace("'seed':config['training_seed'],'signature'","'seed':42,'signature'")
        copied=copied.replace('print(f\'seed={config["training_seed"]} {name}',"print(f'{name}")
        assert copied.strip()==inspect.getsource(original_train_one).strip(), 'Only training RNG/log label may differ'
        assert sorted(p.name for p in (d/'seed_42').iterdir())==['reference.json']
        results=pd.read_csv(d/'per_seed_results.csv')
        geometry=pd.read_csv(d/'per_seed_geometry.csv')
        assert len(results)==len(geometry)==16
        assert set(results.seed)==set(SEEDS)
        new_initials=[]
        for seed in SEEDS:
            hashes=[]
            for name in NAMES:
                directory=ROOT/'outputs/phase212/experiments'/name if seed==42 else d/f'seed_{seed}/experiments'/name
                state=json.loads((directory/'state.json').read_text(encoding='utf-8'))
                log=pd.read_csv(directory/'training_log.csv')
                diag=pd.read_csv(directory/'geometry_log.csv')
                assert list(log.epoch)==list(diag.epoch)==list(range(1,21))
                assert state['complete'] and state['last_epoch']==20
                assert (log.samples_per_epoch==2000).all()
                assert (log.min_anchor_exposure==20).all() and (log.max_anchor_exposure==20).all()
                assert (log.source_forward_rows_per_batch==48).all()
                np.testing.assert_allclose(log.train_total_loss,log.train_triplet_loss+.1*log.train_mean_loss+log.train_variance_loss,rtol=1e-6,atol=1e-7)
                if name==NAMES[0]:
                    assert (log.train_mean_loss==0).all() and (log.train_variance_loss==0).all()
                if name==NAMES[1]:
                    assert (log.train_variance_loss==0).all()
                if name==NAMES[2]:
                    assert (log.train_mean_loss==0).all()
                best=int(log.loc[log.val_competition_score.idxmax(),'epoch'])
                row=results[(results.seed==seed)&(results.experiment==name)].iloc[0]
                assert row.best_epoch==best==state['best_epoch']
                assert file_digest(directory/'best.pt')==state['best_checkpoint_sha256']
                checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
                assert checkpoint['seed']==seed
                assert checkpoint['shared_split_sha256']==reference['split']['file_sha256']
                assert checkpoint['diagnostic_subset_sha256']==reference['split']['diagnostic_subset_sha256']
                assert checkpoint['selection']=='DEV3 normal competition score only'
                hashes.append(checkpoint['initial_model_sha256'])
                for key in ['top1','top3','top5','competition_score']:
                    assert np.isclose(row[key],checkpoint['validation_metrics'][key],rtol=0,atol=1e-15)
                assert (diag.fit_partition=='TRAIN ONLY').all()
                values=np.load(directory/'best_scores.npz',allow_pickle=False)
                assert values['normal'].shape==values['color'].shape==(50,50)
            assert len(set(hashes))==1
            new_initials.append(hashes[0])
        assert len(set(new_initials))==4
        calculated=paired_deltas(results,geometry)
        paired=pd.read_csv(d/'paired_deltas.csv')
        pd.testing.assert_frame_equal(calculated,paired,check_exact=False,rtol=1e-12,atol=1e-15)
        aggregated=aggregate(results,geometry)
        pd.testing.assert_frame_equal(aggregated,pd.read_csv(d/'aggregate_results.csv'),check_exact=False,rtol=1e-12,atol=1e-15)
        assert sign_consistency(calculated,aggregated)==summary['sign_consistency']
        color=pd.read_csv(d/'color_results.csv')
        fusion=pd.read_csv(d/'fusion_results.csv')
        overlap=pd.read_csv(d/'error_overlap.csv')
        assert len(color)==32 and len(fusion)==len(overlap)==16
        assert (fusion.alpha==.9).all() and (fusion.normalization=='z-score').all()
        assert (overlap.questions==50).all()
        assert ((overlap.opening_wrong_triplet_correct+overlap.opening_correct_triplet_wrong+overlap.both_correct+overlap.both_wrong)==50).all()
        mean=pd.read_csv(d/'mean_gradient_diagnostic.csv')
        variance=pd.read_csv(d/'variance_gradient_diagnostic.csv')
        assert len(mean)==4 and len(variance)==16
        assert mean.batch_sha256.nunique()==variance.batch_sha256.nunique()==1
        assert (variance.partition=='TRAIN ONLY').all() and (variance.batch_rows==48).all()
        assert np.isfinite(variance.select_dtypes(include='number').to_numpy()).all()
        assert (variance.variance_gradient_raw_norm>=0).all()
        assert summary['seed42_not_retrained'] and not summary['final_test4_created']
        assert summary['phase212_all_artifacts_unchanged']
        report={'status':'VERIFIED Phase 2.13 DEV RESEARCH ONLY','experiments':16,'new_training_epochs':240,
                'reused_seed42_epochs':80,'same_initialization_within_each_seed':True,
                'different_initialization_between_seeds':True,'rng_only_training_difference':True,
                'all_phase212_artifacts_unchanged':True,'no_new_preprocessing':True,'split_seed':42,
                'best_epoch_normal_only':True,'paired_and_sample_std_verified':True,
                'fixed_train_gradient_batch':True,'no_test_evaluation':True}
        write_json(report,d/'protocol_verification.json')
        print(report)


if __name__=='__main__':
    main()
