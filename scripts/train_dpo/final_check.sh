#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -J nadir-final
#
# The final check of the chosen adapter, in one Slurm job with one vLLM server:
# the test questions of the tuning split, and the Markdown tasks of the
# teammate's "six suites" check.
#   sbatch -o <run_dir>/final_check.log scripts/train_dpo/final_check.sh <run_dir>
# Results: <run_dir>/eval_test.jsonl and .html, <run_dir>/six_suites/<series>/answers.jsonl.
# The test questions are for the one chosen adapter only: checking several
# adapters on them and picking the best would make its figure too good.
#
# Paths are absolute on purpose: Slurm runs a copy of this script from its spool
# directory, so files next to the script cannot be found at run time.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
MODEL="${MODEL:-/data/shared/dpo_team/models/Qwen3-8B}"
TEST="${TEST:-$HOME/data/dpo-qwen/data/tuning_two_kinds/test.jsonl}"
SIX_SUITES_TASKS="${SIX_SUITES_TASKS:-$HOME/data/dpo-qwen/data/six_suites/six_suites_results/04_markdown_base/answers.jsonl}"
# Every job needs its own port: with a shared one a job that starts while another
# is serving talks to that server and gets the answers of the wrong adapter.
PORT="${PORT:-$((31000 + ${SLURM_JOB_ID:-$$} % 4000))}"
RUN="${1:?directory of a training run}"
URL="http://127.0.0.1:$PORT"

export LD_LIBRARY_PATH="$REPO/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

if curl -sf -m 5 -o /dev/null "$URL/health"; then
  echo "final_check.sh: port $PORT is already served by another process" >&2
  exit 1
fi
"$REPO/.venv/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name base \
  --reasoning-parser qwen3 --enable-lora --max-lora-rank 16 --lora-modules "epoch_0=$RUN/epoch_0" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf -o /dev/null "$URL/health" && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "final_check.sh: server exited before becoming healthy" >&2
    exit 1
  fi
  sleep 5
done
curl -sf -o /dev/null "$URL/health" || { echo "final_check.sh: server is not healthy after 15 minutes" >&2; exit 1; }

"$REPO/.venv/bin/python" "$REPO/scripts/train_dpo/eval_holdout.py" \
  --base-url "$URL/v1" --models base epoch_0 --holdout "$TEST" --out "$RUN/eval_test.jsonl"
"$REPO/.venv/bin/python" "$REPO/scripts/train_dpo/six_suites_markdown.py" \
  --base-url "$URL/v1" --tuned epoch_0 --tasks "$SIX_SUITES_TASKS" --out "$RUN/six_suites"
