#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -J qwen-serve
#SBATCH -o qwen-serve_%j.log
#
# Serve Qwen3-8B with an OpenAI-compatible API: sbatch serve.sh
# The server listens on 127.0.0.1:$PORT of the GPU node; see README.md for curl examples.
# Override defaults with: sbatch --export=ALL,PORT=8012 serve.sh
set -euo pipefail

VENV="${VENV:-/home/nsibgatullin/dpo-qwen/.venv}"
MODEL="${MODEL:-/data/shared/dpo_team/models/Qwen3-8B}"
PORT="${PORT:-8011}"

# The conda-forge Python needs its own libstdc++; without LD_LIBRARY_PATH the
# older system one is loaded first and imports fail with "CXXABI_1.3.15 not found".
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Qwen3 thinks by default; disable it so answers come without a <think> block.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name Qwen3-8B \
  --host 127.0.0.1 \
  --port "$PORT" \
  --default-chat-template-kwargs '{"enable_thinking": false}'
