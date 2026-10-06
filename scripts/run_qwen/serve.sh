#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -J qwen-serve
#SBATCH -o qwen-serve_%j.log
#
# Serve Qwen3-8B with SGLang and an OpenAI-compatible API: sbatch serve.sh
# The server listens on 127.0.0.1:$PORT of the GPU node; see README.md for curl examples.
# Override defaults with: sbatch --export=ALL,PORT=30001 serve.sh
#
# Paths are absolute on purpose: Slurm runs a copy of this script from its spool
# directory, so files next to the script cannot be found at run time.
set -euo pipefail

# The mentor's ready-made SGLang environment; readable by every user of the server.
SGLANG_ENV="${SGLANG_ENV:-/home/gpkiryukhov/.conda/envs/sglang}"
COMPAT_DIR="${COMPAT_DIR:-/home/gpkiryukhov/cuda-compat-13-0/compat}"
MODEL="${MODEL:-/data/shared/dpo_team/models/Qwen3-8B}"
PORT="${PORT:-30000}"

# torch and sglang in this environment are built for CUDA 13, the server driver
# (570.x) only supports CUDA 12.8. The forward-compat user-mode driver from
# COMPAT_DIR has to come first in LD_LIBRARY_PATH, otherwise torch sees no GPU.
export CUDA_HOME="$SGLANG_ENV/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$COMPAT_DIR:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CUDA_HOME/bin:$SGLANG_ENV/bin:$PATH"
export CCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1

# triton attention and pytorch sampling avoid the FlashInfer JIT build, which
# fails in this environment (nvcc 13.4 against CUDA 13.0 headers).
exec "$SGLANG_ENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name Qwen3-8B \
  --reasoning-parser qwen3 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --host 127.0.0.1 \
  --port "$PORT" \
  --mem-fraction-static 0.85
