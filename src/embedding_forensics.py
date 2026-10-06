"""Read-only representation capture and TRAIN-only geometry post-processing."""
import numpy as np
import pandas as pd
import torch

LAYERS=['backbone_feature','raw_embedding','normalized_embedding']


def unit(values):
    """Normalize in float64; preserve zero rows explicitly rather than dropping identities."""
    values=np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite representation')
    norms=np.linalg.norm(values,axis=-1,keepdims=True)
    return values/np.where(norms>1e-12,norms,1.)


def capture_representations(model,tensor):
    """Hook the existing forward once, capturing pooled/pre-normalize/final outputs."""
    if model.training:
        raise ValueError('Diagnostics require model.eval(); BatchNorm must not update')
    captured={}
    def before_projection(module,args):
        captured['backbone_feature']=args[0].detach()
    def after_projection(module,args,output):
        captured['raw_embedding']=output.detach()
    hooks=[model.projection.register_forward_pre_hook(before_projection),
           model.projection.register_forward_hook(after_projection)]
    try:
        with torch.no_grad():
            captured['normalized_embedding']=model(tensor).detach()
        return captured
    finally:
        for hook in hooks:
            hook.remove()


def extract_representations(model,store,device,batch_size,directory,partition):
    """Save all position outputs and equal-position game means from a single frozen pass."""
    model.eval()
    total=sum(r['num_positions'] for r in store.records)
    widths={'backbone_feature':model.projection.in_features,
            'raw_embedding':model.projection.out_features,'normalized_embedding':model.projection.out_features}
    directory.mkdir(parents=True,exist_ok=True)
    arrays={name:np.lib.format.open_memmap(directory/(partition+'_'+name+'.npy'),mode='w+',dtype=np.float32,shape=(total,width))
            for name,width in widths.items()}
    pending=[]
    offset=0
    def flush():
        nonlocal offset
        if not pending:
            return
        x=torch.from_numpy(np.stack(pending).astype(np.float32)).to(device)
        representations=capture_representations(model,x)
        for name,value in representations.items():
            arrays[name][offset:offset+len(pending)]=value.cpu().numpy()
        offset+=len(pending)
        pending.clear()
    with torch.inference_mode():
        for index in range(len(store)):
            features=store.features(index)
            if len(features)!=store.records[index]['num_positions']:
                raise ValueError('Position count differs from frozen feature manifest')
            for feature in features:
                pending.append(feature)
                if len(pending)==batch_size:
                    flush()
        flush()
    if offset!=total:
        raise ValueError('Position capture incomplete')
    offsets=np.cumsum([0]+[r['num_positions'] for r in store.records])
    games={name:np.stack([values[a:b].mean(axis=0,dtype=np.float64) for a,b in zip(offsets[:-1],offsets[1:])])
           for name,values in arrays.items()}
    for values in arrays.values():
        values.flush()
    np.savez_compressed(directory/(partition+'_games.npz'),**games)
    print(f'{partition}: captured {len(store)} games / {total} positions at all three layers',flush=True)
    return arrays,games,offsets


def basic_geometry(values,threshold):
    """Report absolute variance and norms without silently unit-normalizing raw/backbone."""
    values=np.asarray(values)
    mean=values.mean(axis=0,dtype=np.float64)
    # Chunk the second moment to bound memory for 64K x 576 arrays.
    variance=np.zeros(values.shape[1],dtype=np.float64)
    norms=[]
    for start in range(0,len(values),2048):
        x=np.asarray(values[start:start+2048],dtype=np.float64)
        variance+=np.square(x-mean).sum(axis=0)
        norms.append(np.linalg.norm(x,axis=1))
    variance/=len(values)
    norm=np.concatenate(norms)
    output={f'variance_{name}':float(value) for name,value in
            [('mean',variance.mean()),('median',np.median(variance)),('min',variance.min()),('max',variance.max())]}
    output['near_zero_fraction']=float(np.mean(variance<threshold))
    output.update({f'norm_{name}':float(value) for name,value in
                   [('mean',norm.mean()),('median',np.median(norm)),('std',norm.std()),('min',norm.min()),('max',norm.max())]})
    mean_norm=float(np.linalg.norm(mean))
    output.update(mean_vector_norm=mean_norm,centered_rms_norm=float(np.sqrt(variance.sum())),
                  common_mean_to_centered_rms=mean_norm/max(np.sqrt(variance.sum()),1e-30))
    return output,variance


def summarize_pairs(cosines,distances,prefix):
    """Summarize one same/different pair class, preserving small float64 differences."""
    return {**{prefix+'_cosine_'+key:float(value) for key,value in
               [('mean',cosines.mean()),('median',np.median(cosines)),('std',cosines.std()),
                ('p25',np.percentile(cosines,25)),('p75',np.percentile(cosines,75))]},
            prefix+'_euclidean_mean':float(distances.mean())}


def game_pair_geometry(candidates,queries,c_labels,q_labels):
    """Compute all candidate/query game pairs exactly, using unsquared Euclidean distance."""
    c=np.asarray(candidates,dtype=np.float64)
    q=np.asarray(queries,dtype=np.float64)
    mask=np.asarray(c_labels)[:,None]==np.asarray(q_labels)[None,:]
    cosine=unit(c)@unit(q).T
    origin=np.concatenate([c,q]).mean(axis=0)
    cc,qq=c-origin,q-origin
    distances=np.sqrt(np.maximum(np.square(cc).sum(axis=1)[:,None]+np.square(qq).sum(axis=1)[None,:]-2*cc@qq.T,0.))
    output={}
    for prefix,selected in [('same',mask),('different',~mask)]:
        if not selected.any():
            raise ValueError('Need both same and different player pairs')
        output.update(summarize_pairs(cosine[selected],distances[selected],prefix))
        output[prefix+'_pair_count']=int(selected.sum())
    output['cosine_separation']=output['same_cosine_mean']-output['different_cosine_mean']
    output['euclidean_separation']=output['different_euclidean_mean']-output['same_euclidean_mean']
    return output


def sample_position_pairs(c_offsets,q_offsets,c_labels,q_labels,count=20000,seed=42):
    """Draw paired position indices deterministically, balanced by same-player identity."""
    rng=np.random.default_rng(seed)
    c_labels=np.asarray(c_labels)
    q_labels=np.asarray(q_labels)
    players=sorted(set(c_labels)&set(q_labels))
    if len(players)<2:
        raise ValueError('Need at least two overlapping candidate/query identities')
    banks={p:(np.flatnonzero(c_labels==p),np.flatnonzero(q_labels==p)) for p in players}
    result={}
    for kind in ['same','different']:
        pairs=[]
        for _ in range(count):
            p=int(rng.integers(len(players)))
            other=p if kind=='same' else (p+int(rng.integers(1,len(players))))%len(players)
            ci=int(rng.choice(banks[players[p]][0]))
            qi=int(rng.choice(banks[players[other]][1]))
            pairs.append((int(rng.integers(c_offsets[ci],c_offsets[ci+1])),int(rng.integers(q_offsets[qi],q_offsets[qi+1]))))
        result[kind]=np.array(pairs,dtype=np.int64)
    return result


def position_pair_geometry(candidates,queries,pairs):
    """Compute sampled pair geometry without a quadratic position-pair matrix."""
    output={}
    for prefix,indices in pairs.items():
        similarities=[]
        distances=[]
        for start in range(0,len(indices),1024):
            ix=indices[start:start+1024]
            c=np.asarray(candidates[ix[:,0]],dtype=np.float64)
            q=np.asarray(queries[ix[:,1]],dtype=np.float64)
            similarities.append(np.einsum('ij,ij->i',unit(c),unit(q)))
            distances.append(np.linalg.norm(c-q,axis=1))
        output.update(summarize_pairs(np.concatenate(similarities),np.concatenate(distances),prefix))
        output[prefix+'_pair_count']=len(indices)
    output['cosine_separation']=output['same_cosine_mean']-output['different_cosine_mean']
    output['euclidean_separation']=output['different_euclidean_mean']-output['same_euclidean_mean']
    return output


def spectral_statistics(eigenvalues):
    """Effective rank and participation ratio distinguish anisotropy from small scale."""
    eigenvalues=np.maximum(np.asarray(eigenvalues,dtype=np.float64),0.)
    total=eigenvalues.sum()
    if total<=0:
        return {'effective_rank':0.,'participation_ratio':0.,'top1_explained':0.,'top5_cumulative':0.,'top10_cumulative':0.,'top20_cumulative':0.}
    weights=eigenvalues/total
    positive=weights[weights>0]
    return {'effective_rank':float(np.exp(-np.sum(positive*np.log(positive)))),
            'participation_ratio':float(total**2/np.square(eigenvalues).sum()),
            **{name:float(weights[:n].sum()) for name,n in [('top1_explained',1),('top5_cumulative',5),('top10_cumulative',10),('top20_cumulative',20)]}}


class TrainGeometry:
    """Fit centering, per-color means and PCA from verified TRAIN game embeddings only."""
    def __init__(self,values,records,partition,allowed_train_ids):
        if partition!='train' or not len(values) or len(values)!=len(records):
            raise ValueError('PCA/centering/whitening fitting requires TRAIN only')
        ids={r['group_id'] for r in records}
        if not ids<=set(allowed_train_ids):
            raise ValueError('Fitting identities outside DEV2 TRAIN')
        x=np.asarray(values,dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite TRAIN embeddings')
        self.mean=x.mean(axis=0)
        self.color_means={color:x[np.array([r['color']==color for r in records])].mean(axis=0) for color in ['B','W']}
        if any(not np.isfinite(v).all() for v in self.color_means.values()):
            raise ValueError('TRAIN must include both B/W colors')
        centered=x-self.mean
        covariance=centered.T@centered/len(centered)
        values,vectors=np.linalg.eigh(covariance)
        order=np.argsort(-values,kind='stable')
        self.eigenvalues=np.maximum(values[order],0.)
        self.components=vectors[:,order]
        # Canonicalize eigenvector sign for saved deterministic projections.
        for col in range(self.components.shape[1]):
            axis=np.argmax(np.abs(self.components[:,col]))
            if self.components[axis,col]<0:
                self.components[:,col]*=-1
        self.statistics=spectral_statistics(self.eigenvalues)
        second_moment=covariance+np.outer(self.mean,self.mean)
        uncentered=np.maximum(np.linalg.eigvalsh(second_moment),0.)
        self.statistics.update(uncentered_top1_energy=float(uncentered[-1]/max(uncentered.sum(),1e-30)),
                               mean_norm=float(np.linalg.norm(self.mean)),centered_rms=float(np.sqrt(self.eigenvalues.sum())),
                               mean_direction_first_centered_pc_alignment=float(abs(unit(self.mean)@self.components[:,0])))
        self.training_players=sorted(ids)
        self.fit_games=len(x)

    def transform(self,values,removed_pcs=0,whiten=False,epsilon=1e-5,colors=None):
        """Apply frozen TRAIN statistics to game embeddings; never inspect VAL labels."""
        if not 0<=removed_pcs<=self.components.shape[1] or epsilon<=0:
            raise ValueError('Invalid PCA removal/whitening protocol')
        x=np.asarray(values,dtype=np.float64)
        mean=self.mean if colors is None else np.stack([self.color_means[color] for color in colors])
        x=x-mean
        if removed_pcs:
            axes=self.components[:,:removed_pcs]
            x=x-(x@axes)@axes.T
        if whiten:
            x=(x@self.components)/np.sqrt(np.maximum(self.eigenvalues,epsilon))
        if not np.isfinite(x).all():
            raise ValueError('Whitening numerically unstable; fixed epsilon must not be changed')
        return unit(x)


def aggregate(values,records,order,game_normalize=False):
    """Mean games equally, then L2-normalize each player/question without dropping groups."""
    values=unit(values) if game_normalize else np.asarray(values,dtype=np.float64)
    groups={}
    for value,record in zip(values,records):
        groups.setdefault(record['group_id'],[]).append(value)
    if set(groups)!=set(order):
        raise ValueError('Every method must use the identical complete player/question pool')
    return unit(np.stack([np.mean(groups[key],axis=0) for key in order]))


def retrieve(candidates,queries,c_records,q_records,players,questions,game_normalize=False):
    """Cosine retrieval with identical axes for every geometry diagnostic."""
    return aggregate(queries,q_records,questions,game_normalize)@aggregate(candidates,c_records,players,game_normalize).T


def color_retrieve(candidates,queries,c_records,q_records,players,questions):
    """Use B→B and W→W only; absent color bank is zero, never an opposite-color fallback."""
    def banks(values,records,order,color):
        grouped={}
        for value,record in zip(values,records):
            if record['color']==color:
                grouped.setdefault(record['group_id'],[]).append(value)
        return unit(np.stack([np.mean(grouped[key],axis=0) if key in grouped else np.zeros(values.shape[1]) for key in order]))
    counts=pd.Series([(r['group_id'],r['color']) for r in q_records]).value_counts().to_dict()
    totals={q:sum(counts.get((q,color),0) for color in ['B','W']) for q in questions}
    if not all(totals.values()) or {r['group_id'] for r in c_records}!=set(players) or {r['group_id'] for r in q_records}!=set(questions):
        raise ValueError('Color-aware retrieval cannot change the player/question pool')
    scores=np.zeros((len(questions),len(players)))
    for color in ['B','W']:
        weights=np.array([counts.get((q,color),0)/totals[q] for q in questions])
        scores+=(banks(queries,q_records,questions,color)@banks(candidates,c_records,players,color).T)*weights[:,None]
    return scores


def player_separability(games,records):
    """Use unsquared Euclidean game-to-centroid and distinct centroid-to-centroid distances."""
    ids=sorted({r['group_id'] for r in records})
    labels=np.array([r['group_id'] for r in records])
    centroids=np.stack([games[labels==p].mean(axis=0) for p in ids])
    lookup={p:i for i,p in enumerate(ids)}
    within=float(np.linalg.norm(games-centroids[[lookup[p] for p in labels]],axis=1).mean())
    centered=centroids-centroids.mean(axis=0)
    distances=np.sqrt(np.maximum(np.square(centered).sum(axis=1)[:,None]+np.square(centered).sum(axis=1)[None,:]-2*centered@centered.T,0.))
    between=float(distances[~np.eye(len(ids),dtype=bool)].mean())
    return {'within_mean':within,'between_mean':between,'ratio':between/within if within>0 else float('inf')}
