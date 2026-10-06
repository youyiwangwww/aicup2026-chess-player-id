"""Inspect checkpoint numerics without performing retrieval or model selection."""
from pathlib import Path
import sys
import time

import torch


def main():
    """Count float32 subnormals to diagnose CPU slowdown without accessing TEST."""
    root = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(root / 'outputs/phase26/experiments/A1/last.pt', weights_only=True, map_location='cpu')
    torch.set_num_threads(4)
    total = 0
    for name, tensor in checkpoint['model_state_dict'].items():
        if tensor.dtype != torch.float32:
            continue
        subnormal = ((tensor.abs() > 0) & (tensor.abs() < torch.finfo(tensor.dtype).tiny)).sum().item()
        if subnormal:
            print(name, subnormal)
        total += subnormal
    print('epoch', checkpoint['epoch'], 'subnormal_values', total)
    sys.path.insert(0, str(root))
    from src.player_model import PlayerEncoder
    from src.feature_cache import GameFeatureStore
    store = GameFeatureStore(root / 'outputs/phase26/experiments/A1/train_manifest.json')
    tensor = torch.stack([torch.from_numpy(store.features(index)[0].astype('float32')) for index in range(48)])
    for version in ['fresh', 'trained']:
        torch.manual_seed(42)
        model = PlayerEncoder(**checkpoint['model_config'])
        if version == 'trained':
            model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        started = time.perf_counter()
        embeddings = model(tensor)
        a, p, n = embeddings.split(16)
        loss = torch.nn.TripletMarginLoss(margin=.2)(a, p, n)
        loss.backward()
        small_gradients = sum(int(((p.grad.abs() > 0) & (p.grad.abs() < torch.finfo(torch.float32).tiny)).sum())
                              for p in model.parameters() if p.grad is not None)
        print('version', version, 'train_batch_seconds', time.perf_counter() - started,
              'subnormal_gradients', small_gradients, flush=True)


if __name__ == '__main__':
    main()
