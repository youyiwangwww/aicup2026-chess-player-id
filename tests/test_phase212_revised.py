"""Fixed-objective algebra, source-pool weighting and disjoint DEV3 protocol tests."""
import copy
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.anti_collapse import NAMES,objective,source_regularizers,GAMMA,LAMBDA_MEAN,LAMBDA_VAR,is_best,classify
from src.phase212_split import make_dev3
from src.phase212_guard import dev3_only
from src.run_phase212 import validate_config
from src.utils import ROOT,load_config


class RevisedAntiCollapseTests(unittest.TestCase):
    def pool(self):
        """Make deterministic unit vectors without touching any actual dataset."""
        torch.manual_seed(42)
        return F.normalize(torch.randn(48,128),dim=1)
    def parts(self,name,pool=None):
        pool=self.pool() if pool is None else pool
        return objective(pool[:16],pool[16:32],pool[32:],pool,name)

    def test_A0_exact_original_triplet_loss(self):
        pool=self.pool()
        parts=self.parts(NAMES[0],pool)
        expected=torch.nn.TripletMarginLoss(margin=.2,p=2)(pool[:16],pool[16:32],pool[32:])
        torch.testing.assert_close(parts['total'],expected,rtol=0,atol=0)
        self.assertIs(parts['total'],parts['triplet'])
        self.assertEqual(parts['mean'].item(),0)
        self.assertEqual(parts['variance'].item(),0)

    def test_A1_exact_decomposition(self):
        parts=self.parts(NAMES[1])
        torch.testing.assert_close(parts['total'],parts['triplet']+.1*parts['mean'],rtol=0,atol=0)
        self.assertEqual(parts['variance'].item(),0)

    def test_A2_exact_decomposition(self):
        parts=self.parts(NAMES[2])
        torch.testing.assert_close(parts['total'],parts['triplet']+parts['variance'],rtol=0,atol=0)
        self.assertEqual(parts['mean'].item(),0)

    def test_A3_exact_decomposition(self):
        parts=self.parts(NAMES[3])
        torch.testing.assert_close(parts['total'],parts['triplet']+.1*parts['mean']+parts['variance'],rtol=0,atol=0)

    def test_identical_unit_vectors_mean_penalty_one(self):
        z=torch.zeros(48,128)
        z[:,0]=1
        mean,_=source_regularizers(z)
        self.assertEqual(mean.item(),1.)

    def test_balanced_opposite_vectors_mean_penalty_zero(self):
        z=torch.zeros(48,128)
        z[:24,0]=1
        z[24:,0]=-1
        mean,_=source_regularizers(z)
        self.assertEqual(mean.item(),0.)

    def test_collapsed_variance_penalty_known_value(self):
        z=torch.zeros(48,128)
        z[:,0]=1
        _,variance=source_regularizers(z)
        self.assertAlmostEqual(variance.item(),GAMMA-.01,places=7)

    def test_spread_unit_vectors_have_lower_variance_penalty(self):
        z=torch.cat([torch.eye(128),-torch.eye(128)])
        _,penalty=source_regularizers(z)
        self.assertEqual(penalty.item(),0.)

    def test_gamma_exact_fixed_inverse_sqrt_128(self):
        self.assertEqual(GAMMA,1/math.sqrt(128))

    def test_lambda_mean_fixed_and_config_changes_rejected(self):
        self.assertEqual(LAMBDA_MEAN,.1)
        c=load_config(ROOT/'configs/phase212.yaml')
        c['regularizers']['lambda_mean']=.2
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_lambda_var_fixed_and_config_changes_rejected(self):
        self.assertEqual(LAMBDA_VAR,1.)
        c=load_config(ROOT/'configs/phase212.yaml')
        c['regularizers']['lambda_var']=2.
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_full_source_pool_once_not_selected_hard_negatives(self):
        pool=self.pool()
        repeated=pool[[0]*16]
        parts=objective(pool[:16],pool[16:32],repeated,pool,NAMES[3])
        mean,var=source_regularizers(pool)
        torch.testing.assert_close(parts['mean'],mean,rtol=0,atol=0)
        torch.testing.assert_close(parts['variance'],var,rtol=0,atol=0)
        self.assertFalse(torch.allclose(parts['mean'],torch.cat([pool,repeated]).mean(dim=0).square().sum()))

    def test_regularizer_rejects_validation(self):
        with self.assertRaises(ValueError):
            source_regularizers(self.pool(),'val')
        with self.assertRaises(ValueError):
            objective(*self.pool().split(16),self.pool(),NAMES[3],'val')

    def source(self):
        """Create enough synthetic identities to test fixed allocation and exclusions."""
        return pd.DataFrame([{'player_id':f'p{i:03d}','game_id':f'g{i}_{j}','sgf_content':f'sgf{i}_{j}','rank':'1d','color':'B' if j%2==0 else 'W'}
                             for i in range(165) for j in range(40)])

    def test_train_val_identity_and_all_game_partitions_disjoint(self):
        frames,audit=make_dev3(self.source(),set(),42)
        self.assertEqual((len(frames[0]),len(frames[1]),len(frames[2])),(2000,1500,500))
        self.assertFalse(set(frames[0].player_id)&set(frames[1].player_id))
        self.assertFalse({'player_id','rank'}&set(frames[2].columns))
        self.assertFalse(any(audit['game_id_overlap'].values()))
        self.assertFalse(any(audit['sgf_content_overlap'].values()))

    def test_all_historical_identities_excluded(self):
        excluded={f'p{i:03d}' for i in range(15)}
        frames,audit=make_dev3(self.source(),excluded,42)
        self.assertFalse((set(frames[0].player_id)|set(frames[1].player_id))&excluded)
        self.assertEqual(audit['historical_player_overlap'],0)

    def test_closed_test_paths_rejected_in_training_scope(self):
        with tempfile.TemporaryDirectory() as folder,dev3_only():
            for path in ['outputs/splits/test_candidates.csv','outputs/phase26/final_test2/triplet_scores.npz','outputs/phase210/splits/ground_truth.csv']:
                with self.assertRaises(PermissionError):
                    (Path(folder)/path).open('rb')

    def test_four_experiments_share_one_config_and_deterministic_split(self):
        c=load_config(ROOT/'configs/phase212.yaml')
        validate_config(c)
        self.assertEqual(c['experiments'],NAMES)
        source=self.source()
        a,_=make_dev3(source,set(),42)
        b,_=make_dev3(source,set(),42)
        for x,y in zip(a,b):
            pd.testing.assert_frame_equal(x,y)

    def test_best_epoch_only_normal_score_earliest_tie(self):
        self.assertTrue(is_best(.4,.3))
        self.assertFalse(is_best(.4,.4))
        self.assertFalse(is_best(.3,.4))
        # Secondary color/fusion/geometry values are deliberately absent from this API.
        with self.assertRaises(TypeError):
            is_best(.3,.4,color_score=.9)

    def test_regularizers_have_finite_gradients(self):
        raw=torch.randn(48,128,requires_grad=True)
        pool=F.normalize(raw,dim=1)
        parts=self.parts(NAMES[3],pool)
        parts['total'].backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(raw.grad.abs().sum().item(),0.)

    def test_success_requires_majority_geometry_and_retrieval(self):
        a={'mean_direction_norm':1.,'raw_pc1_explained':.9,'cosine_separation':.01,'between_within_ratio':.3,'competition_score':.4}
        b={**a,'mean_direction_norm':.5,'raw_pc1_explained':.4,'cosine_separation':.1,'competition_score':.5}
        self.assertEqual(classify(b,a)['classification'],'PROMISING REPRESENTATION INTERVENTION')
        b['competition_score']=.3
        self.assertEqual(classify(b,a)['classification'],'GEOMETRY IMPROVED / RETRIEVAL DEGRADED')


if __name__=='__main__':
    unittest.main()
