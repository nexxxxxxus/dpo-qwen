#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -J nadir-overmarked
#
# Collect over-marked answers from an adapter of a training run, in one Slurm job:
# start vLLM with the adapter, run collect_overmarked.py, stop the server.
#   sbatch -o <log> scripts/train_dpo/collect_overmarked.sh <run_dir>/epoch_N [arguments of collect_overmarked.py]
#
# Paths are absolute on purpose: Slurm runs a copy of this script from its spool
# directory, so files next to the script cannot be found at run time.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
MODEL="${MODEL:-/data/shared/dpo_team/models/Qwen3-8B}"
# Every job needs its own port: with a shared one a job that starts while another
# is serving talks to that server and gets the answers of the wrong adapter.
PORT="${PORT:-$((31000 + ${SLURM_JOB_ID:-$$} % 4000))}"
ADAPTER="${1:?directory of an adapter}"
URL="http://127.0.0.1:$PORT"

# vLLM lives in the repository's .venv; without its lib directory in
# LD_LIBRARY_PATH torch fails with "CXXABI_1.3.15 not found".
export LD_LIBRARY_PATH="$REPO/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

if curl -sf -m 5 -o /dev/null "$URL/health"; then
  echo "collect_overmarked.sh: port $PORT is already served by another process" >&2
  exit 1
fi
"$REPO/.venv/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name base \
  --reasoning-parser qwen3 --enable-lora --max-lora-rank 16 --lora-modules "adapter=$ADAPTER" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf -o /dev/null "$URL/health" && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "collect_overmarked.sh: server exited before becoming healthy" >&2
    exit 1
  fi
  sleep 5
done
curl -sf -o /dev/null "$URL/health" || { echo "collect_overmarked.sh: server is not healthy after 15 minutes" >&2; exit 1; }

"$REPO/.venv/bin/python" "$REPO/scripts/train_dpo/collect_overmarked.py" \
  --base-url "$URL/v1" --model adapter "${@:2}"
