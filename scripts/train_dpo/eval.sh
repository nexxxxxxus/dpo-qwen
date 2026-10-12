#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0:40:00
#SBATCH -J nadir-dpo-eval
#
# Compare the base model and the adapters of a training run on the holdout pairs,
# in one Slurm job: start vLLM with the adapters, run eval_holdout.py, stop the server.
#   sbatch -o <run_dir>/eval.log scripts/train_dpo/eval.sh <run_dir> [arguments of eval_holdout.py]
# for example, on the validation questions of the tuning split:
#   ... eval.sh <run_dir> --holdout ~/data/dpo-qwen/data/tuning/validation.jsonl --out <run_dir>/eval_validation.jsonl
# The base model and the adapters are served by the same vLLM, so the only
# difference between the answers is the adapter. Results: <run_dir>/eval.jsonl and .html.
#
# Paths are absolute on purpose: Slurm runs a copy of this script from its spool
# directory, so files next to the script cannot be found at run time.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
MODEL="${MODEL:-/data/shared/dpo_team/models/Qwen3-8B}"
# Every job needs its own port: with a shared one a job that starts while another
# is serving talks to that server and gets the answers of the wrong adapter.
PORT="${PORT:-$((31000 + ${SLURM_JOB_ID:-$$} % 4000))}"
RUN="${1:?directory of a training run}"
URL="http://127.0.0.1:$PORT"

# vLLM lives in the repository's .venv; without its lib directory in
# LD_LIBRARY_PATH torch fails with "CXXABI_1.3.15 not found".
export LD_LIBRARY_PATH="$REPO/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

MODELS=(base)
LORA=()
for dir in "$RUN"/epoch_*; do
  MODELS+=("$(basename "$dir")")
  LORA+=("$(basename "$dir")=$dir")
done

if curl -sf -m 5 -o /dev/null "$URL/health"; then
  echo "eval.sh: port $PORT is already served by another process" >&2
  exit 1
fi
"$REPO/.venv/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name base \
  --reasoning-parser qwen3 --enable-lora --max-lora-rank 16 --lora-modules "${LORA[@]}" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf -o /dev/null "$URL/health" && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "eval.sh: server exited before becoming healthy" >&2
    exit 1
  fi
  sleep 5
done
curl -sf -o /dev/null "$URL/health" || { echo "eval.sh: server is not healthy after 15 minutes" >&2; exit 1; }

"$REPO/.venv/bin/python" "$REPO/scripts/train_dpo/eval_holdout.py" \
  --base-url "$URL/v1" --models "${MODELS[@]}" --out "$RUN/eval.jsonl" "${@:2}"
