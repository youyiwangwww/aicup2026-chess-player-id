"""Verify deterministic CUDA training and the equivalent pooling, using synthetic tensors only."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metric_utils import seed_everything, write_json
from src.phase28_common import DeterministicAdaptiveAverage, encoder
from src.phase28_triplet import hard_negative
from src.utils import load_config


def main():
    """Check mathematical equivalence and a full existing-architecture CUDA optimizer step."""
    seed_everything(42, True, 4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    a = torch.randn(2, 3, 19, 19, dtype=torch.float64, requires_grad=True)
    b = a.detach().cuda().requires_grad_(True)
    expected = torch.nn.AdaptiveAvgPool2d(3)(a)
    actual = DeterministicAdaptiveAverage(3)(b)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-12, atol=1e-12)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient.cuda())
    torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=1e-12, atol=1e-12)
    config = load_config(ROOT/'configs/phase28.yaml')
    model = encoder(config, 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    pool = model(torch.randn(48, 17, 19, 19, device='cuda'))
    aa, pp, _ = pool.split(16)
    labels = torch.arange(16, device='cuda')
    negative, _ = hard_negative(aa, pool, labels, labels.repeat(3))
    loss = torch.nn.TripletMarginLoss(margin=.2)(aa, pp, negative)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    result = {'pool_forward_equivalent': True, 'pool_gradient_equivalent': True,
        'existing_architecture_backward_step_passed': True, 'deterministic': torch.are_deterministic_algorithms_enabled(),
        'loss': loss.item(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0), 'parameters': sum(p.numel() for p in model.parameters())}
    write_json(result, ROOT/'outputs/phase28/cuda_verification.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
