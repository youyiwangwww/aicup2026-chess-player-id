"""Synthetic DEV-only representation and fitting-isolation regression tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from src.embedding_forensics import (TrainGeometry,capture_representations,color_retrieve,
    retrieve,unit,sample_position_pairs,game_pair_geometry)
from src.player_model import PlayerEncoder
from src.phase211_guard import dev_only


class Phase211Tests(unittest.TestCase):
    def model(self):
        """Use the existing encoder on synthetic tensors, without any checkpoint access."""
        torch.manual_seed(42)
        return PlayerEncoder(channels=4,num_blocks=1,embedding_dim=6).eval()

    def train_fit(self):
        """Fit a small genuine TRAIN fixture with both colors and varied dimensions."""
        x=np.random.default_rng(42).normal(size=(12,6))+np.array([10.,0.,0.,0.,0.,0.])
        records=[{'group_id':f't{i//4}','color':'B' if i%2==0 else 'W'} for i in range(12)]
        return TrainGeometry(x,records,'train',{'t0','t1','t2'}),x,records

    def test_production_forward_and_all_buffers_unchanged(self):
        m=self.model()
        x=torch.randn(3,17,19,19)
        original={k:v.clone() for k,v in m.state_dict().items()}
        with torch.no_grad():
            before=m(x)
            reps=capture_representations(m,x)
            after=m(x)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        torch.testing.assert_close(before,reps['normalized_embedding'],rtol=0,atol=0)
        for key,value in original.items():
            torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(m.projection._forward_hooks)
        self.assertFalse(m.projection._forward_pre_hooks)

    def test_raw_is_projection_output_before_normalization(self):
        m=self.model()
        x=torch.randn(3,17,19,19)
        reps=capture_representations(m,x)
        torch.testing.assert_close(reps['raw_embedding'],m.projection(reps['backbone_feature']),rtol=0,atol=0)
        torch.testing.assert_close(reps['normalized_embedding'],torch.nn.functional.normalize(reps['raw_embedding'],dim=1),rtol=0,atol=0)
        self.assertFalse(torch.allclose(reps['raw_embedding'].norm(dim=1),torch.ones(3)))

    def test_normalized_position_norm_is_one(self):
        reps=capture_representations(self.model(),torch.randn(3,17,19,19))
        torch.testing.assert_close(reps['normalized_embedding'].norm(dim=1),torch.ones(3),rtol=1e-6,atol=1e-6)

    def test_pca_fitting_rejects_val_partition_and_identity(self):
        _,x,records=self.train_fit()
        for partition in ['val','val_candidate','val_query','stability','test']:
            with self.assertRaises(ValueError):
                TrainGeometry(x,records,partition,{'t0','t1','t2'})
        with self.assertRaises(ValueError):
            TrainGeometry(x,records,'train',{'unrelated_training_id'})

    def test_center_mean_is_only_train_and_not_validation_distribution(self):
        fit,x,_=self.train_fit()
        np.testing.assert_allclose(fit.mean,x.mean(axis=0),rtol=0,atol=0)
        original=fit.mean.copy()
        val=np.full((4,6),1000.)
        fit.transform(val)
        np.testing.assert_array_equal(fit.mean,original)
        np.testing.assert_allclose(fit.transform(val),unit(val-x.mean(axis=0)))

    def test_whitening_uses_train_covariance_and_fixed_floor(self):
        fit,x,_=self.train_fit()
        covariance=(x-x.mean(axis=0)).T@(x-x.mean(axis=0))/len(x)
        np.testing.assert_allclose(fit.eigenvalues,np.linalg.eigvalsh(covariance)[::-1])
        val=np.ones((3,6))
        expected=unit(((val-fit.mean)@fit.components)/np.sqrt(np.maximum(fit.eigenvalues,1e-5)))
        np.testing.assert_allclose(fit.transform(val,whiten=True,epsilon=1e-5),expected)

    def test_remove_pc_deterministic_and_k0_is_centering(self):
        fit,x,records=self.train_fit()
        other=TrainGeometry(x,records,'train',{'t0','t1','t2'})
        for k in [0,1,2,4]:
            np.testing.assert_array_equal(fit.transform(x,removed_pcs=k),other.transform(x,removed_pcs=k))
        np.testing.assert_array_equal(fit.transform(x,removed_pcs=0),unit(x-x.mean(axis=0)))

    def color_fixture(self):
        """Reverse white banks to expose any accidental cross-color matching."""
        c=np.array([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
        records=[{'group_id':p,'color':color} for p in ['p0','p1'] for color in ['B','W']]
        return c,records

    def test_black_query_matches_black_only(self):
        c,cr=self.color_fixture()
        s=color_retrieve(c,np.array([[1.,0.]]),cr,[{'group_id':'q','color':'B'}],['p0','p1'],['q'])
        np.testing.assert_array_equal(s,np.array([[1.,0.]]))

    def test_white_query_matches_white_only(self):
        c,cr=self.color_fixture()
        s=color_retrieve(c,np.array([[1.,0.]]),cr,[{'group_id':'q','color':'W'}],['p0','p1'],['q'])
        np.testing.assert_array_equal(s,np.array([[0.,1.]]))

    def test_color_query_weights_and_missing_bank_no_fallback(self):
        c,cr=self.color_fixture()
        q=np.array([[1.,0.],[1.,0.],[1.,0.]])
        qr=[{'group_id':'q','color':color} for color in ['B','B','W']]
        s=color_retrieve(c,q,cr,qr,['p0','p1'],['q'])
        np.testing.assert_allclose(s,[[2/3,1/3]])
        s=color_retrieve(c[[0,2,3]],q, [cr[i] for i in [0,2,3]], qr,['p0','p1'],['q'])
        np.testing.assert_allclose(s,[[2/3,1/3]])

    def test_color_specific_means_use_only_train_colors(self):
        fit,x,records=self.train_fit()
        for color in ['B','W']:
            expected=x[[r['color']==color for r in records]].mean(axis=0)
            np.testing.assert_array_equal(fit.color_means[color],expected)
            np.testing.assert_allclose(fit.transform(np.ones((2,6)),colors=[color,color]),unit(np.ones((2,6))-expected))

    def test_all_closed_input_truth_and_matrix_paths_rejected(self):
        paths=['outputs/splits/test_candidates.csv','outputs/splits/test_queries.csv','outputs/splits/test_ground_truth.csv',
               'outputs/results/test_scores.npz','outputs/round1_archive/scores.npz',
               'outputs/phase26/splits/final_test2_candidates.csv','outputs/phase26/splits/final_test2_queries.csv',
               'outputs/phase26/splits/final_test2_ground_truth.csv','outputs/phase26/final_test2/triplet_scores.npz',
               'outputs/phase210/splits/candidates.csv','outputs/phase210/splits/queries.csv','outputs/phase210/splits/ground_truth.csv',
               'outputs/phase210/opening_scores.npz','outputs/phase29/splits/ground_truth.csv']
        with tempfile.TemporaryDirectory() as folder,dev_only():
            for name in paths:
                with self.assertRaises(PermissionError,msg=name):
                    (Path(folder)/name).open('rb')

    def test_identical_questions_required_for_all_aggregation_methods(self):
        c,cr=self.color_fixture()
        q=np.array([[1.,0.],[0.,1.]])
        qr=[{'group_id':'q0','color':'B'},{'group_id':'q1','color':'W'}]
        for func in [retrieve,color_retrieve]:
            s=func(c,q,cr,qr,['p0','p1'],['q0','q1'])
            self.assertEqual(s.shape,(2,2))
            with self.assertRaises(ValueError):
                func(c,q,cr,qr,['p0','p1'],['q0'])

    def test_no_optimizer_backward_or_bn_training(self):
        model=self.model()
        with patch('torch.optim.Optimizer.__init__',side_effect=AssertionError('optimizer forbidden')), \
             patch('torch.Tensor.backward',side_effect=AssertionError('backward forbidden')):
            reps=capture_representations(model,torch.randn(2,17,19,19))
        self.assertTrue(all(not value.requires_grad for value in reps.values()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError):
            capture_representations(model.train(),torch.randn(2,17,19,19))

    def test_position_pairs_deterministic_and_euclidean_stable_under_huge_mean(self):
        offsets=np.array([0,2,4])
        a=sample_position_pairs(offsets,offsets,['p0','p1'],['p0','p1'],12,42)
        b=sample_position_pairs(offsets,offsets,['p0','p1'],['p0','p1'],12,42)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])
        c=np.array([[1e8,0.],[1e8,2.]])
        q=np.array([[1e8,0.],[1e8,2.]])
        stats=game_pair_geometry(c,q,['p0','p1'],['p0','p1'])
        self.assertEqual(stats['same_euclidean_mean'],0.)
        self.assertEqual(stats['different_euclidean_mean'],2.)


if __name__=='__main__':
    unittest.main()
