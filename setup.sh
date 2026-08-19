#!/usr/bin/env bash
# One-command environment setup for the prospective SSM repo.
# Usage:  bash setup.sh          (auto-detects GPU)
#         bash setup.sh cpu      (force CPU-only)
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

if [ "$1" = "cpu" ]; then
    echo ">> installing CPU-only JAX"
    pip install -r requirements.txt
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo ">> NVIDIA GPU detected — installing JAX with CUDA 12"
    pip install -U "jax[cuda12]==0.11.0" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install numpy flax==0.12.8 optax==0.2.8
else
    echo ">> no GPU detected — installing CPU-only JAX"
    pip install -r requirements.txt
fi

python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"
echo ">> sanity: running the ghost demo (numpy only)"
python ghost_demo.py | head -12
echo ">> done. Activate with: source .venv/bin/activate"
