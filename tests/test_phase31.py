"""Structural professor-network tests and MOCK replay order, not exact-feature proof."""
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.aicup_minizero_adapter import adapt_aicup_row
from src.minizero_policy_backend import RandomStructuralBackend,hidden_shape_smoke,validate_checkpoint_sha
from src.policy_move_statistics import extract_move_statistics,replay_pre_move
from src.phase31_guard import train_smoke_only,require_smoke_operation

ROOT=Path(__file__).resolve().parents[1]

class FakeEnv:
    """Only event-order fixture; these arrays are NOT real MiniZero features."""
    def reset(self):self.count=0;self.events=['reset']
    def get_features(self):self.events.append('feature');return np.full((18,19,19),self.count)
    def act(self,action):self.events.append(tuple(action));self.count+=1;return True

class Phase31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backend=RandomStructuralBackend(ROOT/'external_refs/minizero_policydetection')
        cls.features=torch.zeros(1,18,19,19)
        cls.output=cls.backend.forward(cls.features)
    def stats(self,logits=None,action=0):
        return extract_move_statistics(np.zeros(362) if logits is None else logits,action,
            {'game_id':'g','player_id':'p','color':'B','total_move_number':1})
    def record(self,color='B'):
        return adapt_aicup_row({'game_id':'g','color':color,'sgf_content':'(;SZ[19];B[aa];W[ss];B[];W[jj])'})
    def test_logits_shape(self):self.assertEqual(tuple(self.output['policy_logit'].shape),(1,362))
    def test_softmax_sum(self):self.assertTrue(torch.allclose(self.output['policy'].sum(1),torch.ones(1),atol=1e-6))
    def test_probability_lookup(self):
        logits=np.zeros(362);logits[8]=2
        self.assertAlmostEqual(self.stats(logits,8).actual_probability,math.exp(2)/(361+math.exp(2)))
    def test_actual_rank_three(self):
        logits=np.arange(362,dtype=float)
        s=self.stats(logits,359);self.assertEqual(s.actual_rank,3);self.assertTrue(s.actual_top3);self.assertFalse(s.actual_top1)
    def test_tie_action_index(self):self.assertEqual(self.stats(action=7).actual_rank,8)
    def test_topk_highest(self):
        logits=np.zeros(362);logits[25]=3
        s=self.stats(logits,25);self.assertEqual(s.actual_rank,1);self.assertTrue(s.actual_top1 and s.actual_top3 and s.actual_top5)
    def test_uniform_entropy(self):self.assertAlmostEqual(self.stats().entropy,math.log(362),places=12)
    def test_pass_index(self):self.assertTrue(self.stats(action=361).is_pass)
    def test_pre_move_order(self):
        env=FakeEnv();replay_pre_move(self.record(),env,on_target=lambda move,features:env.events.append('forward'))
        self.assertEqual(env.events[:4],['reset','feature','forward',('B','A19')])
    def test_first_move_pre_state(self):
        self.assertEqual(replay_pre_move(self.record(),FakeEnv())[0][1].sum(),0)
    def test_black_only(self):
        rows=replay_pre_move(self.record('B'),FakeEnv());self.assertEqual([m.color for m,f in rows],['B','B'])
    def test_white_only(self):
        rows=replay_pre_move(self.record('W'),FakeEnv());self.assertEqual([m.color for m,f in rows],['W','W'])
        self.assertEqual(rows[0][1][0,0,0],1)
    def test_hidden_hook_cleanup(self):
        result=hidden_shape_smoke(self.backend,self.features)
        self.assertTrue(result['hook_cleanup']);self.assertEqual(result['hidden_shape'],[1,256,19,19]);self.assertEqual(result['spatial_mean_shape'],[1,256])
    def test_random_not_meaningful(self):
        self.assertFalse(self.backend.metadata()['meaningful_policy_statistics'])
        with self.assertRaises(PermissionError):self.backend.require_aggregation()
    def test_bootstrap_no_player_scores(self):
        with self.assertRaises(PermissionError):require_smoke_operation('tiny_bootstrap_player_score')
    def test_closed_test_denied(self):
        with train_smoke_only(),self.assertRaises(PermissionError):open('outputs/phase210/final_test3/queries.csv')
    def test_val_denied(self):
        with train_smoke_only(),self.assertRaises(PermissionError):open('outputs/phase212/splits/val_candidates.csv')
    def test_checkpoint_hash_validation(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'weights.pkl');p.write_bytes(b'not unpickled')
            self.assertEqual(validate_checkpoint_sha(p,hashlib.sha256(p.read_bytes()).hexdigest()),hashlib.sha256(p.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):validate_checkpoint_sha(p,'0'*64)
    def test_cpu_metadata(self):self.assertEqual(self.backend.metadata()['device_type'],'cpu')
    def test_final_test_creation_denied(self):
        with tempfile.TemporaryDirectory() as d:
            with train_smoke_only(),self.assertRaises(PermissionError):Path(d,'final_test4').mkdir()
    def test_underflow_log_probability(self):
        logits=np.zeros(362);logits[0]=-1000
        s=self.stats(logits);self.assertEqual(s.actual_probability,0);self.assertTrue(math.isfinite(s.actual_log_probability))
        self.assertEqual(s.actual_log_probability,s.policy_actual_log_probability)
    def test_pass_gtp(self):
        env=FakeEnv();replay_pre_move(self.record(),env)
        self.assertIn(('B','PASS'),env.events)
    def test_bad_features(self):
        with self.assertRaises(ValueError):self.backend.forward(torch.zeros(1,17,19,19))
    def test_no_optimizer(self):self.assertFalse(hasattr(self.backend,'optimizer'))
    def test_cpu_provenance_saved(self):
        import json
        m=json.loads((ROOT/'outputs/phase31/environment.json').read_text(encoding='utf-8'))
        for field in ['device_type','device_name','torch_version','cuda_version','python_version','os','cpu','ram']:self.assertIn(field,m)
        self.assertEqual(m['device_type'],'cpu')

if __name__=='__main__':unittest.main()
