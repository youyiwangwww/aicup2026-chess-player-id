"""Verify real CUDA execution on the laptop GPU without any experiment data."""
import json
import platform
import time

import torch


def main():
    """Run a small GPU operation and report the exact runtime environment."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this interpreter')
    started = time.perf_counter()
    tensor = torch.ones((64, 64), device='cuda')
    result = tensor @ tensor
    torch.cuda.synchronize()
    if not torch.all(result == 64):
        raise RuntimeError('CUDA matrix multiplication verification failed')
    report = {'python': platform.python_version(), 'pytorch': str(torch.__version__),
              'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
              'gpu_capability': list(torch.cuda.get_device_capability(0)),
              'verification_seconds': time.perf_counter() - started}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
