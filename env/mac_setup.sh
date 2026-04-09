#!/usr/bin/env bash
set -euo pipefail

TVM_REF="878a61105ea4c85f3547fe137a28d0a80b1f0e94"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-.venv}"
TVM_DIR="${TVM_DIR:-3rdparty/tvm}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

select_python() {
  if [ -n "${PYTHON_BIN}" ]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      echo "${PYTHON_BIN} not found."
      echo "Install Python 3.9–3.11 first, for example:"
      echo "  brew install python@3.11"
      exit 1
    fi
    echo "${PYTHON_BIN}"
    return
  fi

  for candidate in python3.11 python3.10 python3.9 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info.major == 3 and 9 <= sys.version_info.minor <= 11 else 1)
PY
      then
        echo "${candidate}"
        return
      fi
    fi
  done

  echo "No supported Python interpreter found."
  echo "Install Python 3.9–3.11 first, for example:"
  echo "  brew install python@3.11"
  exit 1
}

PYTHON_BIN="$(select_python)"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required but not found."
  exit 1
fi

echo "[macOS Setup 1/8] Installing system dependencies..."
brew update
brew install cmake ninja llvm@14 git libomp

echo "[macOS Setup 2/8] Checking Python version..."
"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info.major != 3 or not (9 <= sys.version_info.minor <= 11):
    raise SystemExit(f"Unsupported Python version: {sys.version_info.major}.{sys.version_info.minor}. Please use Python 3.9–3.11.")
PY

echo "[macOS Setup 3/8] Creating virtual environment..."
if [ ! -d "${PROJECT_ROOT}/${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${PROJECT_ROOT}/${VENV_DIR}"
fi
source "${PROJECT_ROOT}/${VENV_DIR}/bin/activate"

echo "[macOS Setup 4/8] Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
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

echo "[macOS Setup 5/8] Cloning TVM..."
if [ -d "${PROJECT_ROOT}/${TVM_DIR}" ]; then
  echo "${PROJECT_ROOT}/${TVM_DIR} already exists, skipping clone."
else
  mkdir -p "$(dirname "${PROJECT_ROOT}/${TVM_DIR}")"
  git clone --recursive https://github.com/apache/tvm "${PROJECT_ROOT}/${TVM_DIR}"
fi

cd "${PROJECT_ROOT}/${TVM_DIR}"
git fetch --all --tags
git checkout "${TVM_REF}"
git submodule update --init --recursive

echo "[macOS Setup 6/8] Configuring build..."
mkdir -p build
if [ ! -f build/config.cmake ]; then
  cp cmake/config.cmake build/config.cmake
fi

LLVM_CONFIG="$(brew --prefix llvm@14)/bin/llvm-config"

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
    return text + "\n" + repl + "\n"

text = replace_or_append(text, "USE_LLVM", "\"${LLVM_CONFIG}\"")
text = replace_or_append(text, "CMAKE_BUILD_TYPE", "RelWithDebInfo")
text = replace_or_append(text, "HIDE_PRIVATE_SYMBOLS", "ON")
text = replace_or_append(text, "USE_CUDA", "OFF")
text = replace_or_append(text, "USE_OPENCL", "OFF")
text = replace_or_append(text, "USE_METAL", "OFF")

cfg.write_text(text)
PY

echo "[macOS Setup 7/8] Building TVM..."
export PATH="$(brew --prefix llvm@14)/bin:${PATH}"
cd build
cmake -G Ninja ..
ninja

echo "[macOS Setup 8/8] Setting up TVM Python environment..."
TVM_PYTHON_DIR="$(cd ../python && pwd)"
TVM_BUILD_DIR="$(cd ../build && pwd)"
ACTIVATE_FILE="${PROJECT_ROOT}/${VENV_DIR}/bin/activate"

BEGIN_MARKER="# >>> TVM >>>"
END_MARKER="# <<< TVM <<<"

python - <<PY
from pathlib import Path
import re

activate = Path("${ACTIVATE_FILE}")
text = activate.read_text()

begin = "${BEGIN_MARKER}"
end = "${END_MARKER}"

block = f"""{begin}
export TVM_HOME="${PROJECT_ROOT}/${TVM_DIR}"
export TVM_LIBRARY_PATH="${TVM_BUILD_DIR}"
export PYTHONPATH="${TVM_PYTHON_DIR}:\${{PYTHONPATH:-}}"
{end}
"""

pattern = re.compile(
    re.escape(begin) + r".*?" + re.escape(end),
    flags=re.DOTALL,
)

if pattern.search(text):
    text = pattern.sub(block, text)
else:
    text = text.rstrip() + "\\n\\n" + block + "\\n"

activate.write_text(text)
PY

export TVM_HOME="${PROJECT_ROOT}/${TVM_DIR}"
export TVM_LIBRARY_PATH="${TVM_BUILD_DIR}"
export PYTHONPATH="${TVM_PYTHON_DIR}:${PYTHONPATH:-}"

echo "=========================================="
echo "Validating installation..."
python - <<'PY'
import numpy as np
import tvm

print("NumPy version:", np.__version__)
print("TVM version:", tvm.__version__)
print("TVM commit:", tvm.support.libinfo().get("GIT_COMMIT_HASH"))
PY
echo "=========================================="
echo "Done."