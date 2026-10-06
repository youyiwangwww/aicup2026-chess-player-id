# Phase 2.6 runtime notes

All reported training experiments use the original Python 3.12.10 / PyTorch 2.14.1 CPU environment and deterministic algorithms. No CPU/CUDA results are mixed for model selection.

An independent CUDA environment was installed after discovering the RTX 5060 Laptop GPU. The CUDA matrix check passed outside the sandbox. The CUDA diagnostic unit suite had one error: `adaptive_avg_pool2d_backward_cuda` has no deterministic implementation. We retained the existing model and strict determinism requirement; the CUDA environment is not used for this experiment. This is an unresolved limitation for future GPU training, rather than a failure in the CPU experiment. The complete CPU suite passes all 52 tests.

Sandbox DLL loading was initially blocked by application control. Verification outside the sandbox succeeded. Pip also warned that its Scripts directory is not on PATH; commands use an explicit interpreter path.

CUDA test details are retained in `.test-tmp/phase26_cuda_tests.log`. CPU test results are in `.test-tmp/phase26_verification.json`. Unit-test fixtures are isolated under `.test-tmp`; no official FINAL TEST 2 evaluation has been executed.
