"""Verify saved artifacts and refusal without reopening any TEST input or scoring."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import final_test3_lock as lock
from src.metric_utils import file_digest,write_json
from src.phase28_common import fuse


def main():
    """Check frozen hashes, matching axes, saved fusion algebra and consumed-slot refusal."""
    d=ROOT/'outputs/phase210'
    receipt=json.loads((d/'final_test3_receipt.json').read_text(encoding='utf-8'))
    protocol=json.loads((d/'preregistered_protocol.json').read_text(encoding='utf-8'))
    assert receipt['status']=='CLOSED TEST' and receipt['evaluation_count']==1
    assert file_digest(d/'preregistered_protocol.json')==receipt['protocol_sha256']
    for path in ['data/training/train_A.csv','configs/phase28_selected.yaml','configs/phase210.yaml',
                 'outputs/phase28/experiments/Triplet-Hard-100/best.pt','src/player_model.py',
                 'src/phase28_common.py','src/phase28_triplet.py','src/phase28_opening.py',
                 'src/phase210_final_test3.py','src/final_test3_lock.py']:
        assert file_digest(path)==protocol['bindings'][path],path
    matrices={name:np.load(d/(name+'_scores.npz')) for name in ['opening','triplet','fusion']}
    try:
        players=matrices['opening']['players'].tolist()
        questions=matrices['opening']['questions'].tolist()
        for matrix in matrices.values():
            assert matrix['scores'].shape==(100,100)
            assert matrix['players'].tolist()==players and matrix['questions'].tolist()==questions
        np.testing.assert_array_equal(matrices['fusion']['scores'],fuse(matrices['opening']['scores'],matrices['triplet']['scores'],.9,'z-score'))
        for name in ['random','opening','triplet','fusion']:
            pred=pd.read_csv(d/(name+'_predictions.csv'),dtype=str)
            assert set(pred.question_id)==set(questions) and set(pred.player_id)<=set(players)
            assert pred.groupby('question_id').size().eq(5).all()
            assert not pred.duplicated(['question_id','player_id']).any()
    finally:
        for matrix in matrices.values():
            matrix.close()
    try:
        lock.require_unconsumed()
    except PermissionError:
        pass
    else:
        raise AssertionError('Second invocation not refused')
    for name in ['candidates.csv','queries.csv','ground_truth.csv']:
        try:
            with (d/'splits'/name).open():
                raise AssertionError('Closed input opened')
        except PermissionError:
            pass
    result={'status':'VERIFIED CLOSED TEST','evaluation_count':1,'all_frozen_hashes_unchanged':True,
            'score_matrix_shape':[100,100],'identical_questions_candidates':True,
            'fusion_algebra_exact':True,'second_invocation_refused':True,'all_three_csv_opens_refused':True,
            'no_inference_or_rescoring':True}
    write_json(result,d/'closed_verification.json')
    print(result)


if __name__=='__main__':
    main()
