"""Paired seed summaries and TRAIN-only gradient diagnostics; no model selection."""
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .anti_collapse import NAMES,source_regularizers,classify

GEOMETRY=['mean_direction_norm','raw_pc1_explained','normalized_effective_rank',
          'cosine_separation','between_within_ratio','variance_mean','near_zero_fraction']
METRICS=['top1','top3','top5','competition_score']


def paired_deltas(results,geometry):
    """Subtract the A0 with the same training seed, requiring unique complete pairs."""
    merged=results.merge(geometry,on=['seed','experiment'],validate='one_to_one')
    rows=[]
    for seed,group in merged.groupby('seed',sort=True):
        if set(group.experiment)!=set(NAMES) or len(group)!=4:
            raise ValueError('Each seed requires all four unique objectives')
        baseline=group[group.experiment==NAMES[0]].iloc[0].to_dict()
        for name in NAMES[1:]:
            row=group[group.experiment==name].iloc[0].to_dict()
            flags=classify(row,baseline)
            rows.append({'seed':int(seed),'experiment':name,
                         'competition_score_delta':row['competition_score']-baseline['competition_score'],
                         **{key+'_delta':row[key]-baseline[key] for key in GEOMETRY},
                         'retrieval_improved':flags['retrieval_improved'],
                         'geometry_majority_improved':flags['geometry_improved'],
                         'geometry_improved_directions':sum(flags[k] for k in
                             ['mean_direction_down','raw_pc1_down','cosine_separation_up','between_within_up'])})
    return pd.DataFrame(rows)


def aggregate(results,geometry):
    """Use sample standard deviation (ddof=1) across training seeds."""
    merged=results.merge(geometry,on=['seed','experiment'],validate='one_to_one')
    rows=[]
    for name in NAMES:
        group=merged[merged.experiment==name]
        row={'experiment':name,'seeds':len(group),'std_ddof':1}
        for key in METRICS+GEOMETRY:
            values=group[key].to_numpy(dtype=float)
            row.update({key+'_mean':float(values.mean()),key+'_std':float(values.std(ddof=1))})
            if key in METRICS:
                row.update({key+'_min':float(values.min()),key+'_max':float(values.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def sign_consistency(paired,aggregated):
    """Apply the user thresholds across four seeds; never select a single-seed winner."""
    rows=[]
    base=float(aggregated[aggregated.experiment==NAMES[0]].competition_score_mean.iloc[0])
    for name in NAMES[1:]:
        group=paired[paired.experiment==name]
        retrieval=int(group.retrieval_improved.sum())
        geometry=int(group.geometry_majority_improved.sum())
        improved=float(aggregated[aggregated.experiment==name].competition_score_mean.iloc[0])>base
        if len(group)!=4:
            raise ValueError('Final stability decision requires exactly four seeds')
        label='NO STABLE IMPROVEMENT'
        if name==NAMES[2] and improved and retrieval>=3 and geometry>=3:
            label='VARIANCE REGULARIZATION STABLE PROMISING'
        elif name==NAMES[3] and improved and retrieval>=3:
            label='COMBINED RETRIEVAL IMPROVEMENT STABLE'
        elif retrieval in (1,2):
            label='SEED-SENSITIVE RESULT'
        elif improved and retrieval>=3:
            label='RETRIEVAL CONSISTENT / GEOMETRY CRITERION NOT MET'
        rows.append({'experiment':name,'retrieval_improved_count':retrieval,
                     'geometry_majority_improved_count':geometry,'seeds':4,
                     'mean_retrieval_improved':improved,'classification':label})
    return rows


def gradient_values(raw,z=None):
    """Compare gradients before/after L2 normalization, with unchanged objective constants."""
    if z is None:
        z=F.normalize(raw,p=2,dim=1)
    mean,var=source_regularizers(z)
    gm_z=torch.autograd.grad(mean,z,retain_graph=True)[0]
    gm_raw=torch.autograd.grad(mean,raw,retain_graph=True)[0]
    gv_raw=torch.autograd.grad(var,raw,retain_graph=True)[0]
    unit=F.normalize(raw.detach(),p=2,dim=1)
    tangent=gm_raw-(gm_raw*unit).sum(dim=1,keepdim=True)*unit
    std=z.detach().std(dim=0,correction=0)
    return {'mean_loss':mean.item(),'mean_gradient_normalized_norm':gm_z.norm().item(),
            'mean_gradient_raw_norm':gm_raw.norm().item(),'mean_gradient_raw_tangential_norm':tangent.norm().item(),
            'variance_loss':var.item(),'variance_gradient_raw_norm':gv_raw.norm().item(),
            'actual_normalized_std_mean':std.mean().item(),'actual_normalized_std_min':std.min().item(),
            'actual_normalized_std_max':std.max().item(),'raw_norm_mean':raw.detach().norm(dim=1).mean().item()}


def complementarity(opening,triplet,fusion,players,questions,truth):
    """Count Top-1 error overlap on exactly the same questions and candidate order."""
    expected=truth.set_index('question_id').player_id.to_dict()
    if len(set(questions))!=len(questions) or set(questions)!=set(expected):
        raise ValueError('Methods must use identical complete question identities')
    def correct(scores):
        if scores.shape!=(len(questions),len(players)):
            raise ValueError('Score matrix axes changed')
        order=np.argsort(-scores,axis=1,kind='stable')[:,0]
        return np.array([players[i]==expected[q] for q,i in zip(questions,order)])
    o,t,f=map(correct,[opening,triplet,fusion])
    return {'questions':len(questions),'opening_wrong_triplet_correct':int((~o&t).sum()),
            'opening_correct_triplet_wrong':int((o&~t).sum()),'both_correct':int((o&t).sum()),
            'both_wrong':int((~o&~t).sum()),'fusion_correct_when_opening_wrong':int((~o&f).sum()),
            'fusion_wrong_when_opening_correct':int((o&~f).sum())}
