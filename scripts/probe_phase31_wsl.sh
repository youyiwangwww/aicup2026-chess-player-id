#!/usr/bin/env bash
# Read-only dependency inspection; optional pinned clone, no package installation.
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/phase31
{
 uname -a
 for tool in git cmake g++ pkg-config; do command -v "$tool" || true; "$tool" --version 2>&1 | head -n 1; done
 python3 - <<'PY'
import importlib.util
print('torch_spec:',importlib.util.find_spec('torch'))
PY
 pkg-config --modversion opencv4 2>&1 || true
 ls /usr/include/boost/version.hpp /usr/local/lib/libale* /usr/include/ale* 2>&1 || true
 find /usr/local /opt -maxdepth 4 -name TorchConfig.cmake -o -name ALEConfig.cmake 2>/dev/null
} > outputs/phase31/wsl_dependency_probe.log 2>&1
cat outputs/phase31/wsl_dependency_probe.log
if [ "${1:-}" = '--clone' ]; then
 if [ ! -d external_refs/minizero_policydetection ]; then
  git clone --no-checkout --single-branch --branch policy https://github.com/b08202011/minizero_policydetection.git external_refs/minizero_policydetection || exit $?
 fi
 git -C external_refs/minizero_policydetection checkout --detach b44b70f53de6cdaa7e2f44a590d541492148a9ad || exit $?
 git -C external_refs/minizero_policydetection rev-parse HEAD HEAD^{tree} > outputs/phase31/source_git_identity.txt
 cat outputs/phase31/source_git_identity.txt
fi
