"""Boundary tests plus explicitly skipped integration checks when binding is absent."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.phase31b_pipeline import (GATE_NAMES, action_to_gtp, audit_real_env,
    audit_replay, bootstrap_eligible, diagnostic, load_binding, phase31b_only)
from src.phase31_guard import require_smoke_operation
from src.aicup_minizero_adapter import adapt_aicup_row
ROOT=Path(__file__).resolve().parents[1]

class Phase31BBoundaries(unittest.TestCase):
    def test_bootstrap_train_only(self):
        with phase31b_only():
            with (ROOT/'outputs/phase212/splits/train.csv').open('rb') as f: self.assertTrue(f.read(1))
    def test_val_denied(self):
        for name in ('val.csv','val_candidates.csv','val_queries.csv'):
            with phase31b_only(),self.assertRaises(PermissionError):
                open(ROOT/'outputs/phase212/splits'/name)
    def test_closed_test_denied(self):
        with phase31b_only(),self.assertRaises(PermissionError):open('outputs/closed_test/data.csv')
    def test_final_test_denied(self):
        with phase31b_only(),self.assertRaises(PermissionError):open('outputs/final_test/data.csv')
    def test_stability_denied(self):
        with phase31b_only(),self.assertRaises(PermissionError):open('outputs/stability/data.csv')
    def test_bootstrap_requires_real_pipeline(self):
        gates={name:'PASS' for name in GATE_NAMES}
        self.assertTrue(bootstrap_eligible(gates,True))
        for name in GATE_NAMES[2:13]:
            for status in ('BLOCKED','NOT_RUN','FAIL','STRUCTURAL_ONLY'):
                self.assertFalse(bootstrap_eligible({**gates,name:status},True))
    def test_missing_cfg_denies_bootstrap(self):
        self.assertFalse(bootstrap_eligible(dict.fromkeys(GATE_NAMES,'PASS'),False))
    def test_bootstrap_no_identity_label(self):
        with self.assertRaises(PermissionError):require_smoke_operation('player_id_loss')
        with self.assertRaises(PermissionError):require_smoke_operation('player_classification')
    def test_aggregation_no_player_id_score(self):
        require_smoke_operation('aggregation_smoke')
        for operation in ('retrieval','player_top1','competition_score','model_selection'):
            with self.assertRaises(PermissionError):require_smoke_operation(operation)
    def test_professor_patch_denied(self):
        with phase31b_only(),self.assertRaises(PermissionError):
            open(ROOT/'external_refs/minizero_policydetection/minizero/environment/go/go.cpp','a')
    def test_phase31_history_immutable(self):
        with phase31b_only(),self.assertRaises(PermissionError):open(ROOT/'outputs/phase31/gates.json','w')
    def test_performance_output_denied(self):
        with phase31b_only(),self.assertRaises(PermissionError):open(ROOT/'outputs/phase31b/player_score.json','w')
    def test_mapping_contract(self):
        self.assertEqual([action_to_gtp(i) for i in (342,360,0,18,180,361)],['A19','T19','A1','T1','K10','PASS'])
    def test_diagnostics_all_planes(self):
        a=np.zeros((18,19,19),np.float32);a[16]=1
        d=diagnostic(a);self.assertEqual(len(d['planes']),18)
        self.assertEqual(d['planes'][16]['sum'],361)
        self.assertEqual(d['sha256'],diagnostic(a.copy())['sha256'])

@unittest.skipUnless(os.environ.get('PHASE31B_REAL_BINDING')=='1',
    'Real MiniZero binding unavailable; integration not executed')
class Phase31BRealBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.py=load_binding(ROOT/'external_refs/minizero_policydetection',ROOT/'outputs/phase31b/feature_only.cfg')
        cls.audit=audit_real_env(cls.py)
    def record(self,color):
        return adapt_aicup_row({'game_id':'integration_fixture','color':color,
            'sgf_content':'(;SZ[19];B[aa];W[ss];B[jj];W[];B[sa];W[as])'})
    def test_real_binding_import(self):self.assertTrue(hasattr(self.py,'TestDataLoader'))
    def test_real_env_feature_shape(self):self.assertEqual(self.audit['semantic_trace'][0]['shape'],[18,19,19])
    def test_coordinate_acceptance(self):self.assertTrue(all(x['env_accepted'] for x in self.audit['coordinates']))
    def test_pass_replay(self):self.assertEqual(self.audit['coordinates'][-1]['action_index'],361)
    def test_pre_move_feature_ordering(self):self.assertEqual(audit_replay(self.record('B'),self.py)[0]['planes'][0]['sum'],0)
    def test_feature_reproducibility(self):self.assertEqual(audit_replay(self.record('B'),self.py),audit_replay(self.record('B'),self.py))
    def test_bw_turn_handling(self):
        self.assertEqual(audit_replay(self.record('W'),self.py)[0]['planes'][17]['sum'],361)
        self.assertEqual(audit_replay(self.record('B'),self.py)[0]['planes'][16]['sum'],361)
    def test_history_plane_transition(self):self.assertEqual(self.audit['semantic_trace'][2]['planes'][2]['sum'],1)

class Phase31BBootstrapIntegration(unittest.TestCase):
    @unittest.skip('No eligible bootstrap: real feature gates blocked and matching architecture cfg absent')
    def test_checkpoint_save_load(self):
        # No substitute checkpoint is created to masquerade as bootstrap evidence.
        import torch
        checkpoint=ROOT/'outputs/phase31b/bootstrap/checkpoint.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.assertTrue(payload['not_professor_pretrained'])
        self.assertTrue(payload['not_player_id_model'])

if __name__=='__main__':unittest.main()
