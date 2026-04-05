#!/usr/bin/env bash
set -euo pipefail

echo "=== RLM Benchmark: H100 Setup ==="

# ── Check GPU ──
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Python ──
PYTHON=${PYTHON:-python3}
if ! $PYTHON --version 2>/dev/null | grep -qE '3\.(11|12|13)'; then
    echo "WARNING: Python 3.11+ recommended. Current: $($PYTHON --version 2>&1)"
fi

# ── Virtual environment ──
VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ── Install dependencies ──
echo ""
echo "Installing benchmark dependencies ..."
pip install -r requirements.txt

# ── Pre-download model weights ──
echo ""
echo "Pre-downloading model weights (this may take a while) ..."
$PYTHON -c "
from huggingface_hub import snapshot_download
import os

cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print(f'HuggingFace cache: {cache}')

for model in ['Qwen/Qwen3-8B', 'mit-oasys/rlm-qwen3-8b-v0.1']:
    print(f'Downloading {model} ...')
    snapshot_download(model, cache_dir=cache)
    print(f'  done.')
"

# ── Smoke test: vLLM import ──
echo ""
echo "Smoke-testing vLLM import ..."
$PYTHON -c "import vllm; print(f'vLLM {vllm.__version__} OK')"

echo ""
echo "=== Setup complete ==="
echo "Activate the env:  source $VENV_DIR/bin/activate"
echo "Next step:         bash serve_model.sh --model Qwen/Qwen3-8B"
