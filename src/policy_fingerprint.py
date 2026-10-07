"""Policy Fingerprint v1 schema and deterministic aggregation of PREEXISTING move statistics."""
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional
import numpy as np

COLORS=('black','white')
PHASES=('opening','middle','late')
BANK_PHASES=(*PHASES,'overall')
METRICS=('actual_probability','actual_rank','negative_log_probability','entropy','top1_probability','probability_gap','rank_normalized')
SUMMARY=('mean','std','median','p25','p75')


def phase_of(total_move_number):
    """Fixed user protocol: opening1..30, middle31..150, late151+; pass counts."""
    if isinstance(total_move_number,bool) or not isinstance(total_move_number,Integral) or total_move_number<1:
        raise ValueError('Move number must be a positive integer')
    return 'opening' if total_move_number<=30 else 'middle' if total_move_number<=150 else 'late'


@dataclass(frozen=True)
class MovePolicyStatistics:
    """Target-side move statistics already computed elsewhere; raw 362-action softmax v1."""
    game_id:str
    player_id:str
    color:str
    total_move_number:int
    actual_probability:Optional[float]=None
    actual_rank:Optional[float]=None
    entropy:Optional[float]=None
    negative_log_probability:Optional[float]=None
    is_pass:bool=False
    target_player_move_number:Optional[int]=None
    actual_move_coordinate:Optional[str]=None
    actual_action_index:Optional[int]=None
    policy_actual_log_probability:Optional[float]=None
    policy_top1_probability:Optional[float]=None
    policy_top1_action:Optional[int]=None


@dataclass(frozen=True)
class PolicyFingerprint:
    """Twelve banks: B/W/combined × phases/overall; missing values represented by None."""
    group_id:str
    values:dict
    level:str
    schema_version:str='policy_fingerprint_v1'
    owner_id:Optional[str]=None


def feature_schema():
    """Return 468 stable numerical feature names; coverage is stored separately."""
    return tuple(f'{color}_{phase}_{stat}_{metric}' for color in (*COLORS,'combined') for phase in BANK_PHASES
                 for metric in METRICS for stat in SUMMARY)+tuple(
                 f'{color}_{phase}_top{k}_rate' for color in (*COLORS,'combined') for phase in BANK_PHASES for k in (1,3,5))+tuple(
                 f'{color}_{phase}_pass_rate' for color in (*COLORS,'combined') for phase in BANK_PHASES)


def _finite(value,low=None,high=None,integer=False):
    """Treat None/NaN as missing; reject infinities and invalid finite domain values."""
    if value is None:
        return None
    if isinstance(value,bool):
        raise ValueError('Boolean is not a policy numeric statistic')
    value=float(value)
    if math.isnan(value):
        return None
    if not math.isfinite(value) or (low is not None and value<low) or (high is not None and value>high) or (integer and not value.is_integer()):
        raise ValueError('Policy statistic outside its defined domain')
    return value


def aggregate_game(moves,game_id=None):
    """Separate target B/W and total-ply phases; never fill an empty phase with a fake zero."""
    moves=list(moves)
    if not moves and game_id is None:
        raise ValueError('Empty game requires explicit game_id')
    identifier=str(game_id) if game_id is not None else moves[0].game_id
    if any(m.game_id!=identifier for m in moves) or len({m.player_id for m in moves})>1:
        raise ValueError('Game aggregation must contain one player and one game')
    if len({m.color.upper() for m in moves})>1:
        raise ValueError('One target player cannot change color within one game')
    prepared=[]
    seen=set()
    for move in moves:
        color=move.color.upper()
        if color not in ('B','W') or move.total_move_number in seen:
            raise ValueError('Invalid color or repeated move statistic')
        seen.add(move.total_move_number)
        phase=phase_of(move.total_move_number)
        p=_finite(move.actual_probability,0.,1.)
        rank=_finite(move.actual_rank,1.,362.,True)
        entropy=_finite(move.entropy,0.,math.log(362)+1e-10)
        nll=_finite(move.negative_log_probability,0.)
        logp=_finite(move.policy_actual_log_probability,high=0.)
        if nll is None and logp is not None:
            nll=-logp
        if nll is None and p is not None:
            nll=-math.log(max(p,1e-12))
        top1=_finite(move.policy_top1_probability,0.,1.)
        if p is not None and top1 is not None and top1<p-1e-12:
            raise ValueError('Top-1 probability cannot be smaller than actual probability')
        prepared.append((move,color,phase,{'actual_probability':p,'actual_rank':rank,
            'entropy':entropy,'negative_log_probability':nll,'top1_probability':top1,
            'probability_gap':max(0.,top1-p) if p is not None and top1 is not None else None,
            'rank_normalized':(rank-1.)/361. if rank is not None else None}))
    prepared.sort(key=lambda item:item[0].total_move_number)
    values={name:None for name in feature_schema()}
    for bank in (*COLORS,'combined'):
        for phase in BANK_PHASES:
            selected=[r for r in prepared if (phase=='overall' or r[2]==phase) and (bank=='combined' or r[1]==bank[0].upper())]
            prefix=f'{bank}_{phase}'
            values[prefix+'_move_count']=len(selected)
            values[prefix+'_valid_game_count']=int(bool(selected))
            values[prefix+'_missing_mask']=not bool(selected)
            for metric in METRICS:
                array=np.array([r[3][metric] for r in selected if r[3][metric] is not None],dtype=np.float64)
                values[prefix+'_'+metric+'_valid_count']=len(array)
                if len(array):
                    stats=[array.mean(),array.std(ddof=0),np.median(array),np.quantile(array,.25),np.quantile(array,.75)]
                    for stat,value in zip(SUMMARY,stats):
                        values[f'{prefix}_{stat}_{metric}']=float(value)
            ranks=[r[3]['actual_rank'] for r in selected if r[3]['actual_rank'] is not None]
            for k in (1,3,5):
                values[f'{prefix}_top{k}_rate']=float(np.mean(np.array(ranks)<=k)) if ranks else None
            values[prefix+'_pass_rate']=float(np.mean([r[0].is_pass for r in selected])) if selected else None
    return PolicyFingerprint(identifier,values,'game',owner_id=moves[0].player_id if moves else None)


def aggregate_player(games,player_id):
    """Give each valid game equal weight per feature; never let long games dominate."""
    games=sorted(list(games),key=lambda game:game.group_id)
    if any(g.level!='game' or g.schema_version!='policy_fingerprint_v1' for g in games) or len({g.group_id for g in games})!=len(games):
        raise ValueError('Expected distinct v1 game fingerprints')
    if any(g.owner_id is not None and str(g.owner_id)!=str(player_id) for g in games):
        raise ValueError('Player aggregation cannot mix identities')
    values={}
    for name in feature_schema():
        observed=[value for g in games if (value:=_finite(g.values[name])) is not None]
        values[name]=float(np.mean(observed)) if observed else None
        values[name+'_valid_game_count']=len(observed)
    for bank in (*COLORS,'combined'):
        for phase in BANK_PHASES:
            prefix=f'{bank}_{phase}'
            values[prefix+'_move_count']=sum(g.values[prefix+'_move_count'] for g in games)
            values[prefix+'_valid_game_count']=sum(g.values[prefix+'_valid_game_count'] for g in games)
            values[prefix+'_missing_mask']=values[prefix+'_valid_game_count']==0
    return PolicyFingerprint(str(player_id),values,'player',owner_id=str(player_id))
