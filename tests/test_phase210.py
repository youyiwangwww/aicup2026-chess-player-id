"""Synthetic one-shot safety tests; never open any real TEST input."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src import final_test3_lock as lock
from src.phase210_final_test3 import validate_config, fixed_scores, NAMES
from src.phase29_validation import make_stability, paired_bootstrap
from src.phase28_common import encoder, rankings
from src.utils import ROOT, load_config
from tests.test_phase28 import FakeStore


class TestPhase210(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.patch=patch.object(lock,'ROOT',self.root)
        self.patch.start()
        lock.directory().mkdir(parents=True)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_successful_reservation_rejects_second_invocation(self):
        with lock.claim_once():
            with self.assertRaises(PermissionError):
                lock.require_unconsumed()
        with self.assertRaises(PermissionError):
            with lock.claim_once():
                self.fail('second invocation entered')

    def test_inference_failure_is_permanently_closed(self):
        with self.assertRaisesRegex(RuntimeError,'partial'):
            with lock.claim_once():
                raise RuntimeError('partial inference failure')
        self.assertTrue((lock.directory()/'FAILED_CLOSED').exists())
        with self.assertRaises(PermissionError):
            lock.require_unconsumed()

    def test_ground_truth_denied_until_post_inference_scoring(self):
        splits=lock.directory()/'splits'
        splits.mkdir()
        truth=splits/'ground_truth.csv'
        truth.write_text('question_id,player_id\nq,p\n')
        with self.assertRaises(PermissionError):
            truth.read_text()
        with lock.stage('PREFLIGHT'):
            self.assertEqual(len(lock.input_digest(truth)),64)
            with self.assertRaises(PermissionError):
                pd.read_csv(truth)
        with lock.claim_once():
            with lock.stage('INFERENCE'),self.assertRaises(PermissionError):
                truth.read_text()
            with lock.stage('SCORING'):
                self.assertEqual(pd.read_csv(truth).player_id.iloc[0],'p')
        with lock.stage('SCORING'),self.assertRaises(PermissionError):
            truth.read_text()

    def test_every_csv_denied_after_consumption(self):
        path=lock.directory()/'splits/candidates.csv'
        path.parent.mkdir()
        path.write_text('player_id\np\n')
        with lock.claim_once():
            self.assertIn('p',path.read_text())
        with self.assertRaises(PermissionError):
            pd.read_csv(path)

    def test_preregistration_inputs_cannot_mutate(self):
        path=lock.directory()/'splits/queries.csv'
        path.parent.mkdir()
        path.write_text('question_id\nq\n')
        (lock.directory()/'preregistered_protocol.json').write_text('{}')
        for action in [lambda:path.write_text('changed'),lambda:path.unlink(),lambda:path.rename(path.with_name('moved.csv'))]:
            with self.assertRaises(PermissionError):
                action()

    def test_initial_atomic_audit_commit_allowed_only_before_registration(self):
        splits=lock.directory()/'splits'
        splits.mkdir()
        temporary=splits/'audit.tmp'
        temporary.write_text('{}')
        with lock.stage('PREPARATION'):
            temporary.replace(splits/'split_audit.json')
        self.assertTrue((splits/'split_audit.json').exists())
        (lock.directory()/'preregistered_protocol.json').write_text('{}')
        with lock.stage('PREPARATION'),self.assertRaises(PermissionError):
            (splits/'split_audit.json').replace(splits/'altered.json')

    def test_all_historical_identities_excluded_and_full_budget(self):
        historical={f'old{i}' for i in range(10)}
        rows=[]
        for p in list(historical)+[f'new{i:03d}' for i in range(105)]:
            rows.extend({'player_id':p,'game_id':f'{p}-{j}','sgf_content':f'sgf-{p}-{j}','color':'B'} for j in range(40))
        c,q,t,a=make_stability(pd.DataFrame(rows),historical,{'players':100,'candidate_games':30,'query_games':10},42)
        self.assertFalse(set(c.player_id)&historical)
        self.assertEqual(a['max_eligible_players'],105)
        self.assertEqual((len(c),len(q),len(t)),(3000,1000,100))
        self.assertFalse({'player_id','rank'}&set(q.columns))
        self.assertFalse(set(c.game_id)&set(q.game_id))

    def test_fixed_alpha_window_checkpoint_epoch_normalization(self):
        c=load_config(ROOT/'configs/phase210.yaml')
        s=load_config(ROOT/'configs/phase28_selected.yaml')
        validate_config(c,s)
        for key,value in [('alpha',.8),('opening_window',10),('color_aware',False),('best_epoch',17),('normalization','min-max')]:
            bad=copy.deepcopy(c)
            bad['fixed'][key]=value
            with self.assertRaises(ValueError):
                validate_config(bad,s)
        bad=copy.deepcopy(s)
        bad['selection']['triplet']['checkpoint']='another.pt'
        with self.assertRaises(ValueError):
            validate_config(c,bad)

    def test_no_optimizer_backward_or_training_during_actual_inference(self):
        import torch
        model=encoder({'model':{'in_channels':17,'channels':4,'num_blocks':1,'embedding_dim':4,'pool_size':3}},'cpu')
        c,q=FakeStore(2),FakeStore(2)
        for record in q.records:
            record['group_id']='q'+record['group_id'][1:]
        with patch('torch.optim.AdamW',side_effect=AssertionError('optimizer forbidden')),patch('torch.Tensor.backward',side_effect=AssertionError('backward forbidden')):
            t,f=fixed_scores(model,c,q,'cpu',4,['p000','p001'],['q000','q001'],np.eye(2))
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(t.shape,f.shape)

    def test_four_methods_same_candidate_and_question_axes(self):
        players=[f'p{i}' for i in range(6)]
        questions=['q0','q1']
        matrices={name:np.random.default_rng(i).normal(size=(2,6)) for i,name in enumerate(NAMES)}
        for scores in matrices.values():
            pred=rankings(scores,players,questions)
            self.assertEqual(set(pred.question_id),set(questions))
            self.assertTrue(set(pred.player_id)<=set(players))
            self.assertTrue(pred.groupby('question_id').size().eq(5).all())

    def test_bootstrap_deterministic_and_paired(self):
        a=pd.DataFrame({'question_id':['q0','q1'],'competition_score':[1.,0.]})
        b=a.assign(competition_score=[1.,.2])
        x,s=paired_bootstrap(a,b,1000,42)
        y,t=paired_bootstrap(a,b,1000,42)
        self.assertEqual(x,y)
        pd.testing.assert_frame_equal(s,t)
        with self.assertRaises(ValueError):
            paired_bootstrap(a,b.assign(question_id=['q0','q2']))


if __name__=='__main__':
    unittest.main()
