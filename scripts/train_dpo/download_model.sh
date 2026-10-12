#!/usr/bin/env bash
# Download the base model Qwen/Qwen3-8B from Hugging Face (16 GB):
#   scripts/train_dpo/download_model.sh
# Run it in tmux so that a dropped connection does not interrupt the download.
# Run scripts/train_dpo/setup_env.sh first: the downloader comes from the environment.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL="${MODEL:-$HOME/data/dpo-qwen/models/Qwen3-8B}"
export HF_HOME="${HF_HOME:-$HOME/data/dpo-qwen/hf_cache}"
export LD_LIBRARY_PATH="$REPO/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$MODEL"
"$REPO/.venv/bin/hf" download Qwen/Qwen3-8B --local-dir "$MODEL"
ls -la "$MODEL"
echo "model is in $MODEL; pass it to the other scripts with: export MODEL=$MODEL"
