"""CPU-only synthetic contracts; no external weights or real data opened."""
import ast
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from src.aicup_minizero_adapter import adapt_aicup_row,sgf_before_move
from src.policy_action_mapping import sgf_to_action,action_to_sgf
from src.policy_fingerprint import MovePolicyStatistics as M,aggregate_game,aggregate_player,feature_schema,phase_of
from src.phase30_guard import cpu_preparation_only,require_cpu_preparation

class Phase30Tests(unittest.TestCase):
    def row(self,content='(;GM[1]SZ[19]KM[6.5];B[aa];W[ss];B[];W[jj])',color='B'):
        return {'game_id':'g','player_id':'p','rank':'A','color':color,'sgf_content':content}
    def move(self,n=1,p=.2,rank=2,color='B',game='g'):
        return M(game,'p',color,n,p,rank,1.)
    def test_move_order(self):
        self.assertEqual([m.color for m in adapt_aicup_row(self.row()).moves],['B','W','B','W'])
    def test_target_white(self):
        r=adapt_aicup_row(self.row(color='W'))
        self.assertEqual([(m.coordinate,m.target_move_number) for m in r.target_moves],[('ss',1),('jj',2)])
    def test_pass_preserved(self):
        r=adapt_aicup_row(self.row())
        self.assertEqual(r.moves[2].action_index,361)
        self.assertIn('B[]',r.sgf_content)
    def test_board_size_rejected(self):
        with self.assertRaises(ValueError):adapt_aicup_row(self.row('(;SZ[13];B[aa])'))
    def test_setup_rejected(self):
        with self.assertRaises(ValueError):adapt_aicup_row(self.row('(;SZ[19]AB[aa];W[bb])'))
    def test_bad_sgf(self):
        with self.assertRaises(ValueError):adapt_aicup_row(self.row('broken'))
    def test_komi_and_no_identity_leak(self):
        r=adapt_aicup_row(self.row())
        self.assertEqual(r.komi,6.5)
        self.assertNotIn('PB[',r.sgf_content)
        self.assertIn('GM[go_19x19]',r.sgf_content)
    def test_before_move_no_future_leak(self):
        r=adapt_aicup_row(self.row())
        self.assertNotIn(';B[',sgf_before_move(r,1))
        self.assertIn(';B[aa]',sgf_before_move(r,2))
        self.assertNotIn('W[ss]',sgf_before_move(r,2))
    def test_corners_center(self):
        for c,a in [('aa',342),('sa',360),('as',0),('ss',18),('jj',180)]:self.assertEqual(sgf_to_action(c),a)
    def test_all_actions_roundtrip(self):
        for a in range(362):self.assertEqual(sgf_to_action(action_to_sgf(a)),a)
    def test_pass_roundtrip(self):
        self.assertEqual(action_to_sgf(sgf_to_action('')),'')
    def test_professor_nonpass_formula(self):
        for a in range(361):self.assertEqual(action_to_sgf(a),chr(97+a%19)+chr(97+18-a//19))
    def test_invalid_mapping(self):
        for c in ['tt','AA','a','zz']:
            with self.assertRaises(ValueError):sgf_to_action(c)
        for a in [-1,362,True,1.5]:
            with self.assertRaises(ValueError):action_to_sgf(a)
    def test_mapping_color_independent(self):
        for color in ['B','W']:self.assertEqual(adapt_aicup_row(self.row(color=color)).moves[0].action_index,342)
    def test_aggregation_deterministic(self):
        rows=[self.move(3,.4),self.move(1,.2)]
        self.assertEqual(aggregate_game(rows).values,aggregate_game(rows[::-1]).values)
    def test_color_separation(self):
        g=aggregate_game([self.move()])
        self.assertEqual(g.values['black_opening_mean_actual_probability'],.2)
        self.assertIsNone(g.values['white_opening_mean_actual_probability'])
    def test_phase_boundaries(self):
        self.assertEqual([phase_of(n) for n in [1,30,31,150,151]],['opening','opening','middle','middle','late'])
    def test_phase_aggregation(self):
        g=aggregate_game([self.move(1,.1),self.move(31,.2),self.move(151,.3)])
        self.assertEqual([g.values[f'black_{p}_mean_actual_probability'] for p in ['opening','middle','late']],[.1,.2,.3])
    def test_empty_phase(self):
        g=aggregate_game([],game_id='empty')
        self.assertTrue(g.values['black_late_missing_mask'])
        self.assertTrue(all(g.values[n] is None for n in feature_schema()))
    def test_nan_missing(self):
        g=aggregate_game([self.move(1,float('nan'),None)])
        self.assertIsNone(g.values['black_opening_mean_actual_probability'])
        self.assertIsNone(g.values['black_opening_top5_rate'])
    def test_invalid_statistics(self):
        for value in [-.1,1.1,float('inf')]:
            with self.assertRaises(ValueError):aggregate_game([self.move(p=value)])
    def test_equal_game_weight(self):
        a=aggregate_game([self.move(p=.2,game='a')])
        b=aggregate_game([self.move(n,.8,game='b') for n in [1,3,5]])
        self.assertAlmostEqual(aggregate_player([a,b],'p').values['black_opening_mean_actual_probability'],.5)
    def test_topk_and_population_std(self):
        g=aggregate_game([self.move(1,.2,1),self.move(3,.4,6)])
        self.assertEqual(g.values['black_opening_top5_rate'],.5)
        self.assertAlmostEqual(g.values['black_opening_std_actual_probability'],.1)
    def test_closed_test_denied(self):
        for path in ['outputs/phase26/final_test2/queries.csv','outputs/phase210/final_test3/queries.csv','outputs/round1_archive/queries.csv']:
            with cpu_preparation_only(),self.assertRaises(PermissionError):open(path)
    def test_gpu_training_process_denied(self):
        with cpu_preparation_only(),self.assertRaises(PermissionError):subprocess.Popen(['never_launch_training'])
    def test_forbidden_actions(self):
        for action in ['training','gpu','inference','new_split','checkpoint_selection']:
            with self.assertRaises(PermissionError):require_cpu_preparation(action)
    def test_final_test_creation_denied(self):
        with tempfile.TemporaryDirectory() as d:
            with cpu_preparation_only(),self.assertRaises(PermissionError):Path(d,'final_test4').mkdir()
    def test_no_network_imports(self):
        for name in ['aicup_minizero_adapter','policy_action_mapping','policy_fingerprint']:
            tree=ast.parse((Path(__file__).parents[1]/'src'/f'{name}.py').read_text(encoding='utf-8'))
            imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
            self.assertFalse(any(n and n.split('.')[0] in ['torch','requests','urllib','subprocess','minizero'] for n in imports))
    def test_complete_schema(self):
        self.assertEqual(len(feature_schema()),468)
        self.assertEqual(len(set(feature_schema())),468)
    def test_player_owner_mismatch(self):
        with self.assertRaises(ValueError):aggregate_player([aggregate_game([self.move()])],'someone_else')
    def test_gap_normalized_rank_and_overall(self):
        m=M('g','p','W',151,.2,362,1.,policy_top1_probability=.4,policy_actual_log_probability=-2.)
        g=aggregate_game([m])
        self.assertAlmostEqual(g.values['white_late_mean_probability_gap'],.2)
        self.assertEqual(g.values['white_overall_mean_rank_normalized'],1.)
        self.assertEqual(g.values['white_late_mean_negative_log_probability'],2.)

if __name__=='__main__':unittest.main()
