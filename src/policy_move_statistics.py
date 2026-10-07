"""Stable per-move diagnostics; no player retrieval or competition-score computation."""
from dataclasses import asdict,dataclass
import math
from numbers import Integral
import numpy as np
from .policy_fingerprint import MovePolicyStatistics,phase_of

@dataclass(frozen=True)
class ExtractedMoveStatistics(MovePolicyStatistics):
    """Extend Phase3.0 schema with explicitly derived per-move fields."""
    actual_log_probability:float=0.
    actual_top1:bool=False
    actual_top3:bool=False
    actual_top5:bool=False
    probability_gap_to_top1:float=0.
    rank_normalized:float=0.
    game_phase:str='opening'

def extract_move_statistics(logits,actual_action_index,metadata):
    """Float64 log-softmax, descending logits and ascending action-index ties."""
    array=np.asarray(logits,dtype=np.float64)
    if array.shape!=(362,) or not np.isfinite(array).all():raise ValueError('Expected362 finite logits')
    if isinstance(actual_action_index,bool) or not isinstance(actual_action_index,Integral) or not 0<=actual_action_index<362:
        raise ValueError('Invalid actual action')
    action=int(actual_action_index)
    shifted=array-array.max()
    logp=shifted-math.log(float(np.exp(shifted).sum()))
    probabilities=np.exp(logp)
    order=np.lexsort((np.arange(362),-array))
    rank=int(np.flatnonzero(order==action)[0])+1
    first=int(order[0])
    p=float(probabilities[action]);top=float(probabilities[first])
    return ExtractedMoveStatistics(game_id=str(metadata['game_id']),player_id=str(metadata['player_id']),
        color=str(metadata['color']).upper(),total_move_number=int(metadata['total_move_number']),
        actual_probability=p,actual_rank=rank,entropy=float(-(probabilities*logp).sum()),
        negative_log_probability=float(-logp[action]),is_pass=action==361,
        target_player_move_number=metadata.get('target_player_move_number'),
        actual_move_coordinate=metadata.get('actual_move_coordinate'),actual_action_index=action,
        policy_actual_log_probability=float(logp[action]),policy_top1_probability=top,policy_top1_action=first,
        actual_log_probability=float(logp[action]),
        actual_top1=rank==1,actual_top3=rank<=3,actual_top5=rank<=5,
        probability_gap_to_top1=max(0.,top-p),rank_normalized=(rank-1)/361.,
        game_phase=phase_of(metadata['total_move_number']))

def replay_pre_move(record,env,max_target_moves=4,on_target=None):
    """Binding interface: reset; read BEFORE actual move; act afterward, including passes."""
    env.reset()
    observed=[]
    for move in record.moves:
        if move.color==record.target_color:
            features=np.asarray(env.get_features(),dtype=np.float32).copy()
            if features.size!=18*19*19:raise ValueError('Binding did not return18planes')
            observed.append((move,features.reshape(18,19,19)))
            if on_target is not None:
                on_target(move,observed[-1][1])  # Network callback runs BEFORE actual act.
        # MiniZero vector<string> expects GTP (A1..T19, skip I), NOT SGF aa..ss.
        index=move.action_index
        x=index%19
        coordinate='PASS' if index==361 else chr(ord('A')+x+(x>=8))+str(index//19+1)
        ok=env.act([move.color,coordinate])
        if ok is False:raise ValueError('Binding rejected recorded move')
        if len(observed)>=max_target_moves:break
    return observed
