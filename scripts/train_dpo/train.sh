#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J nadir-dpo
#
# LoRA DPO for Qwen3-8B on one A100 with torchtune:
#   sbatch -o <output_dir>/slurm.log scripts/train_dpo/train.sh <output_dir> [config overrides]
# for example: ... train.sh ~/data/dpo-qwen/outputs/2026-10-09-nadir-dpo-highlight epochs=1 optimizer.lr=1e-5
# The adapter of each epoch is saved to <output_dir>/epoch_N, metrics to <output_dir>/logs.
#
# Paths are absolute on purpose: Slurm runs a copy of this script from its spool
# directory, so files next to the script cannot be found at run time.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
CONFIG="$REPO/scripts/train_dpo/config.yaml"
DATA_FILE="${DATA_FILE:-$HOME/data/dpo-qwen/data/highlight/train.jsonl}"
# The shared read-only copy of the base model on our server; set MODEL on another machine.
MODEL="${MODEL:-/data/shared/dpo_team/models/Qwen3-8B}"
OUT="${1:?output directory}"

# Keep what the run was made from next to its result.
mkdir -p "$OUT"
cp "$CONFIG" "$OUT/config.yaml"
cp "$DATA_FILE" "$OUT/train_data.jsonl"

# torchtune lives in the repository's .venv; without its lib directory in
# LD_LIBRARY_PATH torch fails with "CXXABI_1.3.15 not found".
cd "$REPO"
export LD_LIBRARY_PATH="$REPO/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=30
exec .venv/bin/tune run lora_dpo_single_device --config "$CONFIG" \
  data_file="$DATA_FILE" output_dir="$OUT" model_dir="$MODEL" "${@:2}"
