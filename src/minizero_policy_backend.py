"""Import the pinned professor network directly; random weights are structural only."""
from abc import ABC,abstractmethod
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import torch

PINNED_COMMIT='b44b70f53de6cdaa7e2f44a590d541492148a9ad'

def validate_checkpoint_sha(path,expected):
    """Check a checkpoint hash BEFORE any deserialization; no pickle execution here."""
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if len(expected)!=64 or actual.lower()!=expected.lower():
        raise ValueError('Checkpoint SHA256 mismatch')
    return actual

def professor_network_class(source):
    """Use a private package namespace for relative imports; never duplicate architecture."""
    directory=Path(source)/'minizero/network/py'
    for name in ['network_unit.py','alphazero_network.py']:
        if not (directory/name).is_file():raise FileNotFoundError(directory/name)
    namespace='_phase31_professor_'+hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
    if namespace not in sys.modules:
        package=types.ModuleType(namespace)
        package.__path__=[str(directory)]
        sys.modules[namespace]=package
    name=namespace+'.alphazero_network'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,directory/'alphazero_network.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name].AlphaZeroNetwork

def structural_architecture(source):
    """Use repo example.cfg for structural sizes, NOT as a matching missing checkpoint cfg."""
    cfg=Path(source)/'example.cfg'
    values={}
    for line in cfg.read_text(encoding='utf-8').splitlines():
        line=line.split('#',1)[0].strip()
        if '=' in line:
            key,value=line.split('=',1);values[key.strip()]=value.strip()
    if values.get('nn_type_name')!='alphazero':raise ValueError('Unsupported example network')
    return {'input_channels':18,'height':19,'width':19,'action_size':362,
        'channels':int(values['nn_num_hidden_channels']),
        'blocks':int(values['nn_num_blocks']),
        'value_hidden_channels':int(values['nn_num_value_hidden_channels']),
        'discrete_value_size':1,'cfg_path':str(cfg),
        'cfg_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),
        'cfg_role':'REPO EXAMPLE FOR STRUCTURAL SMOKE; NOT MATCHING CHECKPOINT CFG'}

class PolicyBackend(ABC):
    """Inference-only common interface; no optimizer or Player-ID retrieval API."""
    @abstractmethod
    def forward(self,features):pass
    @abstractmethod
    def metadata(self):pass
    def extract_logits(self,features):
        """Return all362 action logits, including pass."""
        return self.forward(features)['policy_logit']
    def extract_probabilities(self,features):
        """Return raw362-action softmax; no legal-action masking."""
        return self.forward(features)['policy']

class RandomStructuralBackend(PolicyBackend):
    """STRUCTURAL SMOKE ONLY / NOT POLICY MODEL / NOT RESEARCH RESULT."""
    def __init__(self,source,seed=42):
        self.device=torch.device('cpu')
        self.architecture=structural_architecture(source)
        a=self.architecture
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network=professor_network_class(source)('go_19x19',18,19,19,a['channels'],19,19,
                a['blocks'],362,a['value_hidden_channels'],1).to(self.device).eval()
        self.seed=seed
    def forward(self,features):
        """Only CPU eval/inference mode; require exact18-plane shape and finite input."""
        features=torch.as_tensor(features,dtype=torch.float32,device=self.device)
        if features.ndim!=4 or tuple(features.shape[1:])!=(18,19,19) or features.shape[0]<1 or not torch.isfinite(features).all():
            raise ValueError('Expected finite (B,18,19,19) features')
        with torch.inference_mode():return self.network(features)
    def metadata(self):
        """Explicitly prohibit interpretation of random logits as meaningful policy."""
        return {'backend':'RandomStructuralBackend','random_weights':True,
            'meaningful_policy_statistics':False,'research_feature':False,
            'label':'STRUCTURAL SMOKE ONLY / NOT POLICY MODEL / NOT RESEARCH RESULT',
            'source_commit':PINNED_COMMIT,'device_type':'cpu','seed':self.seed,
            'architecture':self.architecture}
    def require_aggregation(self):
        """Random model outputs cannot become research fingerprints."""
        raise PermissionError('Random backend cannot aggregate Player Fingerprints')

class ProfessorCheckpointBackend(PolicyBackend):
    """Future contract only; audited matching cfg/source/SHA required before implementation."""
    def __init__(self,*args,**kwargs):
        raise RuntimeError('Verified professor checkpoint/cfg not available; loading disabled')
    def forward(self,features):raise RuntimeError('Unavailable verified backend')
    def metadata(self):return {'available':False}

VerifiedProfessorBackend=ProfessorCheckpointBackend

def hidden_shape_smoke(backend,features):
    """Temporary last-residual hook; cleanup guaranteed even on forward failure."""
    captured=[]
    block=backend.network.residual_blocks[-1]
    before=len(block._forward_hooks)
    handle=block.register_forward_hook(lambda module,args,out:captured.append(out.detach()))
    try:
        backend.forward(features)
        tensor=captured[0]
        result={'hidden_shape':list(tensor.shape),'spatial_mean_shape':list(tensor.mean(dim=(2,3)).shape)}
    finally:handle.remove()
    result['hook_cleanup']=len(block._forward_hooks)==before
    return result
