"""Read-only TRAIN monitoring and complete best-checkpoint DEV3 geometry."""
import numpy as np
import torch
from .embedding_forensics import capture_representations,basic_geometry,spectral_statistics,game_pair_geometry,player_separability,unit


class GameSubset:
    """Reference a frozen list of TRAIN game IDs without altering shared feature caches."""
    def __init__(self,store,game_ids):
        wanted=set(game_ids)
        self.store=store
        self.indices=[i for i,r in enumerate(store.records) if r['game_id'] in wanted]
        self.records=[store.records[i] for i in self.indices]
        if {r['game_id'] for r in self.records}!=wanted:
            raise ValueError('TRAIN diagnostic subset not fully available')
    def __len__(self):
        return len(self.records)
    def features(self,index):
        """Retrieve exactly the original frozen sampled positions for each selected game."""
        return self.store.features(self.indices[index])


def capture_games(model,store,device,batch_size=64):
    """Capture raw/unit positions once and aggregate their game means equally."""
    model.eval()
    pending,counts=[],[r['num_positions'] for r in store.records]
    chunks={'raw_embedding':[],'normalized_embedding':[]}
    def flush():
        if not pending:
            return
        x=torch.from_numpy(np.stack(pending).astype(np.float32)).to(device)
        values=capture_representations(model,x)
        for name in chunks:
            chunks[name].append(values[name].cpu().numpy())
        pending.clear()
    with torch.inference_mode():
        for i in range(len(store)):
            for position in store.features(i):
                pending.append(position)
                if len(pending)==batch_size:
                    flush()
        flush()
    positions={name:np.concatenate(values) for name,values in chunks.items()}
    offsets=np.cumsum([0]+counts)
    games={name:np.stack([values[a:b].mean(axis=0,dtype=np.float64) for a,b in zip(offsets[:-1],offsets[1:])]) for name,values in positions.items()}
    return positions,games


def covariance_stats(values):
    """Report raw PC concentration or normalized effective rank, never transform VAL data."""
    values=np.asarray(values,dtype=np.float64)
    centered=values-values.mean(axis=0)
    spectrum=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(values))[::-1],0.)
    return spectral_statistics(spectrum)


def train_monitor(model,subset,device,batch_size,threshold):
    """Compute fixed TRAIN position-level diagnostics in eval mode, without VAL fitting."""
    positions,_=capture_games(model,subset,device,batch_size)
    z,r=positions['normalized_embedding'],positions['raw_embedding']
    geometry,_=basic_geometry(z,threshold)
    raw,_=basic_geometry(r,threshold)
    return {'mean_direction_norm':geometry['mean_vector_norm'],
            **{k:geometry[k] for k in ['variance_mean','variance_median','variance_min','variance_max','near_zero_fraction']},
            'raw_mean_vector_norm':raw['mean_vector_norm'],'raw_mean_norm':raw['norm_mean'],
            'raw_centered_rms':raw['centered_rms_norm'],'raw_mean_scatter_ratio':raw['common_mean_to_centered_rms'],
            'raw_pc1_explained':covariance_stats(r)['top1_explained'],
            'normalized_effective_rank':covariance_stats(z)['effective_rank'],
            'diagnostic_games':len(subset),'diagnostic_positions':len(z),'fit_partition':'TRAIN ONLY'}


def best_geometry(c_raw,q_raw,c_norm,q_norm,c_records,q_records,truth,threshold):
    """Measure full VAL unit game directions and raw game covariance after epoch selection."""
    z=np.concatenate([unit(c_norm),unit(q_norm)])
    raw=np.concatenate([c_raw,q_raw])
    stats,variance=basic_geometry(z,threshold)
    mapping=dict(zip(truth.question_id,truth.player_id))
    pairs=game_pair_geometry(c_norm,q_norm,[r['group_id'] for r in c_records],[mapping[r['group_id']] for r in q_records])
    separability=player_separability(unit(c_norm),c_records)
    return {**stats,**pairs,'mean_direction_norm':stats['mean_vector_norm'],
            'raw_pc1_explained':covariance_stats(raw)['top1_explained'],
            'normalized_effective_rank':covariance_stats(z)['effective_rank'],
            'within_mean':separability['within_mean'],'between_mean':separability['between_mean'],
            'between_within_ratio':separability['ratio']},variance
