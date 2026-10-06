"""Verify saved DEV fitting provenance and immutable frozen artifacts without inference."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase211_guard import dev_only,closed_markers
from src.metric_utils import file_digest,write_json
from src.utils import load_config


def main():
    """Check fitting inputs/axes/forward hashes and deny actual CLOSED-path open attempts."""
    with dev_only():
        d=ROOT/'outputs/phase211'
        s=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        selected=load_config(ROOT/'configs/phase28_selected.yaml')
        old=json.loads((ROOT/'outputs/phase28/protocol_verification.json').read_text(encoding='utf-8'))
        for name in ['src/player_model.py','src/phase28_common.py','src/phase28_triplet.py','configs/phase28_selected.yaml']:
            assert file_digest(name)==old['code_config_sha256'][name],name
        assert file_digest('outputs/phase28/experiments/Triplet-Hard-100/best.pt')==s['checkpoint_sha256']
        for part,key in [('train','metric_train_csv'),('val_candidate','val_candidates_csv'),('val_query','val_queries_csv'),('truth','val_ground_truth_csv')]:
            assert file_digest(selected['paths'][key])==s['split_sha256'][part]
        records=json.loads((d/'representations/train_records.json').read_text(encoding='utf-8'))
        train=pd.read_csv(ROOT/selected['paths']['metric_train_csv'],dtype=str,usecols=['game_id','player_id','color'])
        assert {r['game_id'] for r in records}==set(train.game_id)
        assert {r['group_id'] for r in records}==set(train.player_id)
        assert len(records)==4000 and len(set(train.player_id))==200
        with np.load(d/'representations/train_games.npz') as games:
            for layer in ['raw_embedding','normalized_embedding']:
                with np.load(d/(layer+'_train_statistics.npz')) as fit:
                    np.testing.assert_array_equal(fit['mean'],games[layer].mean(axis=0))
                    for color,key in [('B','mean_black'),('W','mean_white')]:
                        mask=np.array([r['color']==color for r in records])
                        np.testing.assert_array_equal(fit[key],games[layer][mask].mean(axis=0))
                    assert set(fit['training_players'])==set(train.player_id)
                    centered=games[layer]-fit['mean']
                    expected=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(centered))[::-1],0.)
                    np.testing.assert_allclose(expected,fit['eigenvalues'],rtol=1e-10,atol=1e-18)
        score_files=list((d/'scores').glob('*.npz'))
        assert len(score_files)==15
        for path in score_files:
            with np.load(path) as matrix:
                assert matrix['scores'].shape==(100,100) and np.isfinite(matrix['scores']).all()
                assert matrix['players'].tolist()==s['players'] and matrix['questions'].tolist()==s['questions']
        pca=pd.read_csv(d/'pca_diagnostics.csv')
        assert all(f'eigenvalue_{rank}' in pca.columns for rank in range(1,21))
        blocked=[]
        for path in ['outputs/splits/test_candidates.csv','outputs/splits/test_queries.csv','outputs/splits/test_ground_truth.csv',
                     'outputs/results/test_scores.npz','outputs/phase26/splits/final_test2_candidates.csv',
                     'outputs/phase26/splits/final_test2_queries.csv','outputs/phase26/splits/final_test2_ground_truth.csv',
                     'outputs/phase26/final_test2/triplet_scores.npz','outputs/phase210/splits/candidates.csv',
                     'outputs/phase210/splits/queries.csv','outputs/phase210/splits/ground_truth.csv',
                     'outputs/phase210/triplet_scores.npz','outputs/phase29/splits/ground_truth.csv']:
            try:
                with (ROOT/path).open('rb'):
                    raise AssertionError('Forbidden path opened')
            except PermissionError:
                blocked.append(path)
        report={'status':'VERIFIED DEV-ONLY FORENSICS','closed_markers':closed_markers(),
                'blocked_path_checks':len(blocked),'production_code_config_checkpoint_unchanged':True,
                'train_only_center_color_means_pca_verified':True,'identical_axes_all_methods':True,
                'full_pca_top20_schema':True,'no_inference_training_or_closed_file_read':True}
        write_json(report,d/'protocol_verification.json')
        print(report)


if __name__=='__main__':
    main()
