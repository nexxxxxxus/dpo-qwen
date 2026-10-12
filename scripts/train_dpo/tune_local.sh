#!/usr/bin/env bash
# The same grid as scripts/train_dpo/tune.sh for a machine without Slurm: runs
# one after another on one GPU, each followed by its evaluation on the
# validation questions.
#   GPU=0 scripts/train_dpo/tune_local.sh                # in tmux; takes hours
#   LRS="2e-5 5e-5" BETAS="0.1 0.3" CHECKPOINTS=1 scripts/train_dpo/tune_local.sh
# A run that already has its validation answers is skipped, so the script can be
# started again after an interruption. Do not use it on our shared server: there
# GPUs are given out by Slurm only.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
DATA="${DATA:-$HOME/data/dpo-qwen/data/tuning_two_kinds}"
OUTPUTS="${OUTPUTS:-$HOME/data/dpo-qwen/outputs}"
LRS="${LRS:-1e-5 2e-5 5e-5}"
BETAS="${BETAS:-0.05 0.1 0.3}"
CHECKPOINTS="${CHECKPOINTS:-4}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250}"
export MODEL="${MODEL:-$HOME/data/dpo-qwen/models/Qwen3-8B}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

for lr in $LRS; do
  for beta in $BETAS; do
    name="tune-lr$lr-beta$beta"
    out="$OUTPUTS/$(date +%F)-$name"
    if ls "$OUTPUTS"/*-"$name"/eval_validation.jsonl >/dev/null 2>&1; then
      echo "$(date +%T) lr $lr, beta $beta: already evaluated, skipping"
      continue
    fi
    mkdir -p "$out"
    echo "$(date +%T) lr $lr, beta $beta: training, log $out/slurm.log"
    # The adapter is saved at the end of an "epoch", so an epoch is cut to CHECKPOINT_STEPS steps.
    if ! DATA_FILE="$DATA/train.jsonl" bash "$REPO/scripts/train_dpo/train.sh" "$out" \
      optimizer.lr="$lr" loss.beta="$beta" lr_scheduler.num_warmup_steps=50 \
      epochs="$CHECKPOINTS" max_steps_per_epoch="$CHECKPOINT_STEPS" > "$out/slurm.log" 2>&1; then
      echo "$(date +%T) lr $lr, beta $beta: TRAINING FAILED, see $out/slurm.log"
      continue
    fi
    echo "$(date +%T) lr $lr, beta $beta: evaluation, log $out/eval.log"
    bash "$REPO/scripts/train_dpo/eval.sh" "$out" --holdout "$DATA/validation.jsonl" \
      --out "$out/eval_validation.jsonl" > "$out/eval.log" 2>&1 ||
      echo "$(date +%T) lr $lr, beta $beta: EVALUATION FAILED, see $out/eval.log"
  done
done
echo "$(date +%T) done; table: $REPO/.venv/bin/python $REPO/scripts/train_dpo/tune_table.py --runs '*tune-*'"
