"""Real binding checks only. No replacement feature extractor or identity evaluation."""
import hashlib
import importlib
from pathlib import Path
import numpy as np
from .phase31_guard import train_smoke_only, require_smoke_operation
from .policy_move_statistics import replay_pre_move
from .policy_action_mapping import sgf_to_action
from contextlib import contextmanager
import contextvars
import os
import sys

_active=contextvars.ContextVar('phase31b_active',default=False)

def _audit(event,args):
    if not _active.get() or event not in ('open','os.mkdir','os.remove','os.rename'): return
    for name in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(name,(str,bytes,Path)): continue
        path=Path(os.fsdecode(name)).resolve().as_posix().lower()
        if any(s in path for s in ('final_test','closed_test','stability')):
            raise PermissionError('Phase 3.1B denies TEST and Stability')
        if '/outputs/phase212/splits/' in path and Path(path).name!='train.csv':
            raise PermissionError('Phase 3.1B permits DEV3 TRAIN only')
        mutation=event!='open' or any(c in (args[1] or '') for c in 'wax+') or (args[2] or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if mutation and '/external_refs/minizero_policydetection/' in path and '/build/' not in path:
            raise PermissionError('SOURCE_PATCH_REQUIRED; professor source is immutable')
        if mutation and '/outputs/phase31/' in path:
            raise PermissionError('Phase 3.1 history is immutable')
        if mutation and '/outputs/phase31b/' in path and any(s in Path(path).name for s in ('retrieval','player_score','competition','fusion')):
            raise PermissionError('No Player-ID performance artifacts')

sys.addaudithook(_audit)

@contextmanager
def phase31b_only():
    with train_smoke_only():
        token=_active.set(True)
        try: yield
        finally: _active.reset(token)

def action_to_gtp(action):
    if isinstance(action,bool) or not isinstance(action,(int,np.integer)) or not 0<=action<=361:
        raise ValueError('Expected action 0..361')
    x=int(action)%19
    return 'PASS' if action==361 else chr(ord('A')+x+(x>=8))+str(int(action)//19+1)

GATE_NAMES = ['container runtime','professor container','C++ build','binding import',
    'Env reset','real 18-plane features','coordinate agreement','pre-move ordering',
    'Black replay','White replay','pass','history planes','reproducibility',
    'real network forward','bootstrap training','checkpoint save/load',
    'move statistics','fingerprint aggregation engineering']

def bootstrap_eligible(gates, architecture_verified):
    """A structural or mock pass can never authorize training."""
    required = GATE_NAMES[2:13]
    return architecture_verified and all(gates.get(k) == 'PASS' for k in required)

def feature_array(env):
    array=np.asarray(env.get_features(),dtype=np.float32)
    if array.size != 18*19*19 or not np.isfinite(array).all():
        raise ValueError('Real binding features must be finite 18x19x19')
    return array.reshape(18,19,19).copy()

def diagnostic(array):
    return {'shape':list(array.shape),'finite':bool(np.isfinite(array).all()),
        'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
        'planes':[{'plane':i,'sum':float(p.sum()),'nonzero':int(np.count_nonzero(p))}
                  for i,p in enumerate(array)]}

def load_binding(source, cfg):
    """Use compiled binding; missing artifacts cause an error, never a fake Env."""
    import sys
    source=Path(source).resolve()
    sys.path.insert(0,str(source))
    try:
        py=importlib.import_module('build.go.minizero_py')
        if not Path(py.__file__).resolve().is_relative_to(source/'build/go'):
            raise RuntimeError('Binding imported from unexpected source')
        for name in ('load_config_file','Env','DataLoader','TestDataLoader'):
            if not hasattr(py,name): raise RuntimeError('Missing binding attribute: '+name)
        if not py.load_config_file(str(Path(cfg).resolve())):
            raise RuntimeError('Feature-only cfg rejected')
        return py
    finally:
        sys.path.remove(str(source))

def audit_real_env(py):
    """Verify source semantics against real arrays on a capture-free sequence.

    Expected masks below are assertions for this tiny sequence, not a feature
    extractor and never model inputs. Planes 0..15 use the current turn's
    own/opponent perspective for each of the last eight post-action boards.
    """
    env=py.Env();env.reset()
    trace=[];history=[];board={'B':set(),'W':set()};turn='B'
    sequence=[('B','aa'),('W','ss'),('B','jj'),('W',''),('B','sa'),('W','as')]
    for step in range(len(sequence)+1):
        array=feature_array(env)
        expected=np.zeros((18,361),dtype=np.float32)
        expected[16 if turn=='B' else 17]=1
        for age,state in enumerate(reversed(history[-8:])):
            for offset,color in enumerate((turn,'W' if turn=='B' else 'B')):
                for action in state[color]: expected[2*age+offset,action]=1
        if not np.array_equal(array.reshape(18,361),expected):
            raise RuntimeError('SOURCE SEMANTICS MISMATCH: stop before training')
        trace.append({'before_move':step+1,'turn':turn,**diagnostic(array)})
        if step == len(sequence): break
        color,coordinate=sequence[step];action=sgf_to_action(coordinate)
        if env.act([color,action_to_gtp(action)]) is not True:
            raise RuntimeError('Coordinate rejected: '+coordinate)
        if action!=361: board[color].add(action)
        history.append({c:set(stones) for c,stones in board.items()})
        turn='W' if color=='B' else 'B'
    # Independently verify all corners, center and pass from a fresh board.
    coordinates=[]
    for coordinate in ('aa','sa','as','ss','jj',''):
        env.reset();action=sgf_to_action(coordinate);gtp=action_to_gtp(action)
        accepted=env.act(['B',gtp])
        array=feature_array(env)
        if accepted is not True: raise RuntimeError('COORDINATE MISMATCH: stop')
        if action!=361 and (array.reshape(18,361)[1,action]!=1 or array[1].sum()!=1):
            raise RuntimeError('COORDINATE MISMATCH: accepted at wrong position')
        coordinates.append({'sgf':coordinate,'gtp':gtp,'action_index':action,'env_accepted':True})
    return {'semantic_trace':trace,'coordinates':coordinates,'real_features_obtained':True}

def audit_replay(record, py):
    """Existing replay_pre_move, with binding instrumentation and exact repeat."""
    def once():
        class Observed:
            def __init__(self): self.env=py.Env();self.acts=0;self.last_read=None
            def reset(self):self.env.reset();self.acts=0
            def get_features(self):self.last_read=self.acts;return self.env.get_features()
            def act(self,action):
                accepted=self.env.act(action)
                self.acts+=1
                return accepted
        env=Observed();rows=[]
        def callback(move,features):
            if env.last_read!=move.total_move_number-1 or env.acts!=move.total_move_number-1:
                raise RuntimeError('Actual move acted before feature read')
            if not np.isfinite(features).all(): raise ValueError('Nonfinite real feature')
            rows.append({'game_id':record.game_id,'move_number':move.total_move_number,
                'color':move.color,'sgf':move.coordinate,'gtp':action_to_gtp(move.action_index),
                'action_index':move.action_index,**diagnostic(features)})
        tensors=replay_pre_move(record,env,max_target_moves=6,on_target=callback)
        for row in rows: row['env_accepted']=True
        return tensors,rows
    with phase31b_only():
        first,rows=once();second,_=once()
    if len(first)!=len(second) or not all(np.array_equal(a[1],b[1]) for a,b in zip(first,second)):
        raise RuntimeError('Feature reproducibility failed')
    return rows
