#!/usr/bin/env bash
# Submit a grid of DPO training runs to Slurm: every learning rate with every beta.
# Each run starts from the base model, saves an adapter every CHECKPOINT_STEPS
# steps and is followed by an evaluation of all its adapters on the validation
# questions. Run on the server, not through sbatch:
#   scripts/train_dpo/tune.sh
#   LRS="1e-5 2e-5" BETAS="0.1" scripts/train_dpo/tune.sh     # another grid
#   CHECKPOINTS=1 TRAIN_TIME=1:00:00 EVAL_TIME=0:30:00 scripts/train_dpo/tune.sh  # short runs: 250 steps each
# Results: ~/data/dpo-qwen/outputs/<date>-nadir-tune-lr<lr>-beta<beta>/, one
# table for all runs: scripts/train_dpo/tune_table.py.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
# The split without reworded pairs: with them the model learned to shorten its answers.
DATA="${DATA:-$HOME/data/dpo-qwen/data/tuning_two_kinds}"
OUTPUTS="${OUTPUTS:-$HOME/data/dpo-qwen/outputs}"
LRS="${LRS:-1e-5 2e-5 5e-5}"
BETAS="${BETAS:-0.05 0.1 0.3}"
CHECKPOINTS="${CHECKPOINTS:-4}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250}"
# Time limit of one training job; a run takes about half an hour per checkpoint.
TRAIN_TIME="${TRAIN_TIME:-3:30:00}"
EVAL_TIME="${EVAL_TIME:-1:00:00}"

for lr in $LRS; do
  for beta in $BETAS; do
    name="nadir-tune-lr$lr-beta$beta"
    out="$OUTPUTS/$(date +%F)-$name"
    mkdir -p "$out"
    # The adapter is saved at the end of an "epoch", so an epoch is cut to CHECKPOINT_STEPS steps.
    train=$(sbatch --parsable -t "$TRAIN_TIME" -J "$name" -o "$out/slurm.log" \
      --export=ALL,DATA_FILE="$DATA/train.jsonl" "$REPO/scripts/train_dpo/train.sh" "$out" \
      optimizer.lr="$lr" loss.beta="$beta" lr_scheduler.num_warmup_steps=50 \
      epochs="$CHECKPOINTS" max_steps_per_epoch="$CHECKPOINT_STEPS")
    eval=$(sbatch --parsable -t "$EVAL_TIME" -J "$name-eval" -o "$out/eval.log" --dependency="afterok:$train" \
      "$REPO/scripts/train_dpo/eval.sh" "$out" --holdout "$DATA/validation.jsonl" --out "$out/eval_validation.jsonl")
    echo "lr $lr, beta $beta: training job $train, evaluation job $eval, $out"
  done
done
