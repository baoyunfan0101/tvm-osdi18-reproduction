#!/usr/bin/env bash
set -euo pipefail

TVM_REF="878a61105ea4c85f3547fe137a28d0a80b1f0e94"
TVM_DIR="${TVM_DIR:-3rdparty/tvm}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[Colab Setup 1/9] Checking Python version..."
python - <<'PY'
import sys
if sys.version_info.major != 3 or not (9 <= sys.version_info.minor <= 11):
    raise SystemExit(
        f"Unsupported Python version: {sys.version_info.major}.{sys.version_info.minor}. "
        "Please use Python 3.9–3.11."
    )
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY

echo "[Colab Setup 2/9] Checking GPU..."
nvidia-smi

echo "[Colab Setup 3/9] Installing system dependencies..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  curl \
  unzip \
  libtinfo-dev \
  zlib1g-dev \
  libedit-dev \
  libxml2-dev \
  llvm-14 \
  llvm-14-dev \
  llvm-14-tools \
  clang-14

echo "[Colab Setup 4/9] Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y apache-tvm tvm numpy || true
python -m pip install \
  "numpy<2" \
  attrs \
  cloudpickle \
  decorator \
  ml-dtypes \
  psutil \
  scipy \
  tornado \
  typing-extensions

echo "[Colab Setup 5/9] Preparing TVM source tree..."
rm -rf "${PROJECT_ROOT:?}/${TVM_DIR}"
mkdir -p "$(dirname "${PROJECT_ROOT}/${TVM_DIR}")"
git clone --recursive https://github.com/apache/tvm "${PROJECT_ROOT}/${TVM_DIR}"

cd "${PROJECT_ROOT}/${TVM_DIR}"
git fetch --all --tags
git checkout "${TVM_REF}"
git submodule sync --recursive
git submodule update --init --recursive

echo "[Colab Setup 6/9] Configuring TVM build..."
rm -rf build
mkdir -p build
cp cmake/config.cmake build/config.cmake

LLVM_CONFIG="$(command -v llvm-config-14 || true)"
if [ -z "${LLVM_CONFIG}" ]; then
  echo "llvm-config-14 not found."
  exit 1
fi

python - <<PY
from pathlib import Path
import re

cfg = Path("build/config.cmake")
text = cfg.read_text()

def replace_or_append(text, key, value):
    pattern = rf'^\s*set\({key}\s+.*?\)\s*$'
    repl = f'set({key} {value})'
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, repl, text, flags=re.MULTILINE)
    return text.rstrip() + "\n" + repl + "\n"

text = replace_or_append(text, "USE_LLVM", "\"${LLVM_CONFIG}\"")
text = replace_or_append(text, "USE_CUDA", "ON")
text = replace_or_append(text, "USE_OPENCL", "OFF")
text = replace_or_append(text, "USE_METAL", "OFF")
text = replace_or_append(text, "USE_VULKAN", "OFF")
text = replace_or_append(text, "USE_ROCM", "OFF")
text = replace_or_append(text, "USE_RPC", "ON")
text = replace_or_append(text, "USE_GRAPH_EXECUTOR", "ON")
text = replace_or_append(text, "USE_PROFILER", "ON")
text = replace_or_append(text, "CMAKE_BUILD_TYPE", "RelWithDebInfo")
text = replace_or_append(text, "HIDE_PRIVATE_SYMBOLS", "ON")

cfg.write_text(text)
PY

echo "[Colab Setup 7/9] Building TVM..."
cd build
cmake -G Ninja ..
ninja

echo "[Colab Setup 8/9] Installing TVM Python package..."
cd ../python
export TVM_HOME="${PROJECT_ROOT}/${TVM_DIR}"
export TVM_LIBRARY_PATH="$(cd ../build && pwd)"
rm -rf build dist ./*.egg-info
python -m pip install --no-build-isolation .

echo "[Colab Setup 9/9] Validating installation..."
python - <<PY
import numpy as np
import tvm

print("NumPy version:", np.__version__)
print("TVM version:", tvm.__version__)
print("TVM commit:", tvm.support.libinfo().get("GIT_COMMIT_HASH"))
print("CUDA exist:", tvm.cuda(0).exist)

if not tvm.cuda(0).exist:
    raise SystemExit("TVM was built, but CUDA is still unavailable.")
PY

echo "=========================================="
echo "Done."