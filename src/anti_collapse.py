"""Four fixed TRAIN-only objectives; regularize the source forward pool exactly once."""
import math
import torch
from torch.nn import functional as F

NAMES=['A0-Triplet','A1-Triplet-Mean','A2-Triplet-Variance','A3-Triplet-MeanVariance']
LAMBDA_MEAN=.1
LAMBDA_VAR=1.
GAMMA=1/math.sqrt(128)
EPSILON=1e-4


def source_regularizers(pool,partition='train'):
    """Use population variance of every source-forward row, before hard-negative reuse."""
    if partition!='train' or pool.ndim!=2 or pool.shape[1]!=128:
        raise ValueError('Regularizers accept only 128-dimensional TRAIN source embeddings')
    mean=pool.mean(dim=0)
    std=torch.sqrt(pool.var(dim=0,correction=0)+EPSILON)
    return mean.square().sum(),F.relu(GAMMA-std).mean()


def objective(anchor,positive,negative,source_pool,experiment,partition='train'):
    """Return unweighted components and their exact fixed-weight total; A0 is unchanged."""
    if experiment not in NAMES or partition!='train':
        raise ValueError('Only the four fixed TRAIN objectives are permitted')
    triplet=F.triplet_margin_loss(anchor,positive,negative,margin=.2,p=2)
    zero=triplet.new_zeros(())
    mean,var=source_regularizers(source_pool,partition) if experiment!=NAMES[0] else (zero,zero)
    mean=mean if experiment in [NAMES[1],NAMES[3]] else zero
    var=var if experiment in [NAMES[2],NAMES[3]] else zero
    total=triplet if experiment==NAMES[0] else triplet+LAMBDA_MEAN*mean+LAMBDA_VAR*var
    return {'total':total,'triplet':triplet,'mean':mean,'variance':var}


def is_best(score,previous):
    """Select only normal DEV3 score; equal scores retain the earliest epoch."""
    return float(score)>float(previous)


def classify(row,baseline):
    """Require three of four geometry directions, separately from retrieval improvement."""
    signs={'mean_direction_down':row['mean_direction_norm']<baseline['mean_direction_norm'],
           'raw_pc1_down':row['raw_pc1_explained']<baseline['raw_pc1_explained'],
           'cosine_separation_up':row['cosine_separation']>baseline['cosine_separation'],
           'between_within_up':row['between_within_ratio']>baseline['between_within_ratio']}
    geometric=sum(signs.values())>=3
    retrieval=row['competition_score']>baseline['competition_score']
    label=('PROMISING REPRESENTATION INTERVENTION' if geometric and retrieval else
           'GEOMETRY IMPROVED / RETRIEVAL DEGRADED' if geometric and not retrieval else
           'RETRIEVAL IMPROVED WITHOUT CLEAR GEOMETRY FIX' if retrieval else
           'NO JOINT IMPROVEMENT')
    return {**signs,'geometry_improved':geometric,'retrieval_improved':retrieval,'classification':label,
            'geometry_notes':f'{sum(signs.values())}/4 primary geometry directions improved'}
