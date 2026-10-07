"""Multi-seed protocol invariants and pure TRAIN-gradient/paired-analysis checks."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from src.anti_collapse import NAMES,objective,is_best,GAMMA,EPSILON,LAMBDA_MEAN,LAMBDA_VAR
from src.run_phase213 import SEEDS,config_for_seed,validate_settings,train_one
from src.run_phase212 import model_digest
from src.phase213_analysis import GEOMETRY,paired_deltas,aggregate,gradient_values,complementarity,sign_consistency
from src.phase213_guard import research_only
from src.phase28_common import encoder
from src.phase28_triplet import ExposureTriplets
from src.metric_utils import seed_everything,file_digest
from src.utils import ROOT,load_config


def sample_tables():
    """Provide shuffled, seed-paired data with known variance and improvement signs."""
    rows,geo=[],[]
    for seed,base in zip(SEEDS,[.1,.2,.3,.4]):
        for i,name in enumerate(NAMES):
            rows.append({'seed':seed,'experiment':name,'top1':base,'top3':base,'top5':base,
                         'competition_score':base+.01*i})
            values=dict(zip(GEOMETRY,[1-.01*i,.9-.01*i,90+i,.01*i,1+.1*i,.001,1.]))
            geo.append({'seed':seed,'experiment':name,**values})
    return pd.DataFrame(rows).sample(frac=1,random_state=42),pd.DataFrame(geo)


class MultiSeedTests(unittest.TestCase):
    def test_split_seed_fixed(self):
        config=load_config(ROOT/'configs/phase213.yaml')
        validate_settings(config)
        config['split_seed']=123
        with self.assertRaises(ValueError):
            validate_settings(config)

    def test_training_seed_does_not_change_split_hash_or_cache_seed(self):
        base=load_config(ROOT/'configs/phase212.yaml')
        audit=json.loads((ROOT/'outputs/phase212/splits/split_audit.json').read_text(encoding='utf-8'))
        for seed in SEEDS[1:]:
            c=config_for_seed(seed)
            self.assertEqual(c['paths'],base['paths'])
            self.assertEqual(c['seed'],42)
            self.assertEqual(c['split'],base['split'])
            for path,digest in audit['file_sha256'].items():
                self.assertEqual(file_digest(ROOT/'outputs/phase212/splits'/path),digest)

    def test_within_seed_identical_initialization(self):
        torch.set_num_threads(1)
        config=config_for_seed(123)
        hashes=[]
        for _ in NAMES:
            seed_everything(123,True,1)
            hashes.append(model_digest(encoder(config,torch.device('cpu'))))
        self.assertEqual(len(set(hashes)),1)

    def test_different_seed_initialization(self):
        hashes=[]
        config=config_for_seed(123)
        for seed in SEEDS:
            seed_everything(seed,True,1)
            hashes.append(model_digest(encoder(config,torch.device('cpu'))))
        self.assertEqual(len(set(hashes)),4)

    def test_seed42_cannot_train(self):
        with self.assertRaises(ValueError):
            config_for_seed(42)

    def test_objective_settings_unchanged(self):
        self.assertEqual((LAMBDA_MEAN,LAMBDA_VAR,GAMMA,EPSILON),(.1,1.,1/np.sqrt(128),1e-4))
        base=load_config(ROOT/'configs/phase212.yaml')
        for seed in SEEDS[1:]:
            c=config_for_seed(seed)
            for key in ['model','training','regularizers','features','fusion','diagnostics']:
                self.assertEqual(c[key],base[key])

    def test_best_only_normal_score(self):
        self.assertTrue(is_best(.4,.3))
        self.assertFalse(is_best(.4,.4))
        source=inspect.getsource(train_one)
        self.assertIn("is_best(values['competition_score'],best)",source)
        self.assertNotIn('is_best(color',source)

    def test_paired_same_seed(self):
        r,g=sample_tables()
        paired=paired_deltas(r,g)
        for i,name in enumerate(NAMES[1:],1):
            np.testing.assert_allclose(paired[paired.experiment==name].competition_score_delta,.01*i)
        with self.assertRaises(ValueError):
            paired_deltas(r.iloc[:-1],g)

    def test_aggregate_mean_sample_std(self):
        r,g=sample_tables()
        a=aggregate(r,g)
        self.assertAlmostEqual(a.iloc[0].competition_score_mean,.25)
        self.assertAlmostEqual(a.iloc[0].competition_score_std,np.std([.1,.2,.3,.4],ddof=1))
        self.assertEqual(a.iloc[0].competition_score_min,.1)
        self.assertEqual(a.iloc[0].competition_score_max,.4)

    def test_collapsed_mean_raw_tangent_gradient(self):
        unit=torch.ones(128,dtype=torch.float64)/np.sqrt(128)
        raw=(3*unit).repeat(48,1).requires_grad_()
        v=gradient_values(raw)
        self.assertAlmostEqual(v['mean_loss'],1.)
        self.assertGreater(v['mean_gradient_normalized_norm'],.1)
        self.assertLess(v['mean_gradient_raw_norm'],1e-12)
        self.assertLess(v['mean_gradient_raw_tangential_norm'],1e-12)

    def test_variance_gradient_finite_nonzero(self):
        raw=torch.randn(48,128,generator=torch.Generator().manual_seed(42),dtype=torch.float64,requires_grad=True)
        v=gradient_values(raw)
        self.assertTrue(np.isfinite(list(v.values())).all())
        self.assertGreater(v['variance_gradient_raw_norm'],0.)
        self.assertGreater(v['actual_normalized_std_mean'],0.)

    def test_closed_test_paths_rejected_before_open(self):
        with research_only():
            for p in ['outputs/round1_archive/candidates.csv','outputs/phase26/final_test2/ground_truth.csv',
                      'outputs/phase210/splits/ground_truth.csv']:
                with self.assertRaises(PermissionError):
                    (ROOT/p).read_bytes()

    def test_no_new_identities_or_reference_mutation(self):
        with research_only():
            for p in ['outputs/phase213/splits/new.csv','outputs/phase213/features/new.npz',
                      'outputs/phase212/summary.json']:
                with self.assertRaises(PermissionError):
                    (ROOT/p).open('w')

    def test_no_final_test4(self):
        with research_only():
            with self.assertRaises(PermissionError):
                (ROOT/'outputs/phase213/final_test4.csv').open('w')

    def test_training_seed_controls_epoch_and_position_sampling(self):
        class Store:
            records=[{'group_id':str(p),'color':'B','num_positions':16} for p in range(3) for _ in range(3)]
        a=ExposureTriplets(Store(),20,123)
        b=ExposureTriplets(Store(),20,123)
        c=ExposureTriplets(Store(),20,2026)
        x=[a.sample_indices(i) for i in range(30)]
        self.assertEqual(x,[b.sample_indices(i) for i in range(30)])
        self.assertNotEqual(x,[c.sample_indices(i) for i in range(30)])
        a.set_epoch(1)
        self.assertNotEqual(x,[a.sample_indices(i) for i in range(30)])

    def test_complementarity_identical_axes_and_counts(self):
        truth=pd.DataFrame({'question_id':['q1','q2'],'player_id':['a','b']})
        o=np.array([[0.,1.],[0.,1.]])
        t=np.array([[1.,0.],[1.,0.]])
        v=complementarity(o,t,t,['a','b'],['q1','q2'],truth)
        self.assertEqual(v['opening_wrong_triplet_correct'],1)
        self.assertEqual(v['opening_correct_triplet_wrong'],1)
        self.assertEqual(v['fusion_correct_when_opening_wrong'],1)
        with self.assertRaises(ValueError):
            complementarity(o,t,t,['a','b'],['q1','q3'],truth)

    def test_four_seed_stability_threshold(self):
        r,g=sample_tables()
        labels=sign_consistency(paired_deltas(r,g),aggregate(r,g))
        self.assertEqual(labels[1]['classification'],'VARIANCE REGULARIZATION STABLE PROMISING')
        self.assertEqual(labels[2]['classification'],'COMBINED RETRIEVAL IMPROVEMENT STABLE')


if __name__=='__main__':
    unittest.main()
