#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J qwen-bench
#
# Run the benchmarks on the base model and on LoRA adapters in one go: start vLLM
# with all of them, ask the questions model after model, stop the server, print
# the table.
#   bash run_bench.sh                      # without Slurm; pick the card with CUDA_VISIBLE_DEVICES
#   sbatch -o bench.log run_bench.sh       # with Slurm
# Started again, it continues: saved answers are not asked for twice.
#
# What is run is set by variables:
#   MODEL     the base model, a folder with config.json and safetensors
#   VENV      the environment with vLLM (scripts/train_dpo/setup_env.sh + scripts/bench/setup_bench.sh)
#   ADAPTERS  "name=folder name=folder ..."; a folder holds adapter_model.safetensors.
#             Empty: every <RUNS>/*tune-lr<lr>-beta<beta>/epoch_0, named lr<lr>-beta<beta>
#   RUNS      where the training runs are, for the search above
#   BASE=0    do not ask the base model
#   Anything after the script name goes to run_bench.py, e.g. --limit 20 or --only ifeval.
set -euo pipefail
# Under Slurm the script runs as a copy from the spool directory, so the folder is
# the one sbatch was called from; otherwise it is the folder of the script.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  BENCH_DIR="${BENCH_DIR:-$SLURM_SUBMIT_DIR}"
else
  BENCH_DIR="${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
fi
[ -f "$BENCH_DIR/run_bench.py" ] || { echo "run_bench.sh: run_bench.py is not in $BENCH_DIR; set BENCH_DIR or call sbatch from the folder of the scripts" >&2; exit 1; }

VENV="${VENV:-$HOME/dpo-qwen/.venv}"
MODEL="${MODEL:-$HOME/data/dpo-qwen/models/Qwen3-8B}"
RUNS="${RUNS:-$HOME/data/dpo-qwen/outputs}"
# In the archive the questions and the results lie next to the scripts.
if [ -d "$BENCH_DIR/data/benchmarks" ]; then
  DATA="${DATA:-$BENCH_DIR/data/benchmarks}"
  RESULTS="${RESULTS:-$BENCH_DIR/results}"
else
  DATA="${DATA:-$HOME/data/dpo-qwen/data/benchmarks}"
  RESULTS="${RESULTS:-$HOME/data/dpo-qwen/outputs/bench}"
fi
# Every run needs its own port: with a shared one a job that starts while another
# is serving talks to that server and gets the answers of the wrong adapters.
PORT="${PORT:-$((31000 + ${SLURM_JOB_ID:-$$} % 4000))}"
URL="http://127.0.0.1:$PORT"

export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NLTK_DATA="${NLTK_DATA:-$BENCH_DIR/nltk_data}"
export PYTHONUNBUFFERED=1

if [ -z "${ADAPTERS:-}" ]; then
  ADAPTERS=""
  for dir in "$RUNS"/*tune-lr*-beta*/epoch_0; do
    [ -f "$dir/adapter_model.safetensors" ] || continue
    # ...-tune-lr1e-5-beta0.05/epoch_0 -> lr1e-5-beta0.05; folders with anything after beta are other experiments
    name=$(basename "$(dirname "$dir")" | grep -o -E "lr[0-9e.-]+-beta[0-9.]+$") || continue
    ADAPTERS="$ADAPTERS $name=$dir"
  done
fi
MODELS=()
[ "${BASE:-1}" = "0" ] || MODELS+=(base)
LORA=()
for pair in $ADAPTERS; do
  MODELS+=("${pair%%=*}")
  LORA+=("$pair")
done
[ "${#MODELS[@]}" -gt 0 ] || { echo "run_bench.sh: nothing to run: no adapters found in $RUNS and BASE=0" >&2; exit 1; }
echo "models: ${MODELS[*]}"
echo "questions: $DATA, results: $RESULTS"

if curl -sf -m 5 -o /dev/null "$URL/health"; then
  echo "run_bench.sh: port $PORT is already served by another process" >&2
  exit 1
fi
SERVE=("$VENV/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name base --reasoning-parser qwen3)
[ "${#LORA[@]}" -eq 0 ] || SERVE+=(--enable-lora --max-lora-rank 16 --lora-modules "${LORA[@]}")
mkdir -p "$RESULTS"
"${SERVE[@]}" > "$RESULTS/vllm_$PORT.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf -o /dev/null "$URL/health" && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "run_bench.sh: the server exited before becoming healthy, see $RESULTS/vllm_$PORT.log" >&2
    exit 1
  fi
  sleep 5
done
curl -sf -o /dev/null "$URL/health" || { echo "run_bench.sh: the server is not healthy after 15 minutes, see $RESULTS/vllm_$PORT.log" >&2; exit 1; }

# The exit code is that of the benchmark run, not of stopping the server.
status=0
"$VENV/bin/python" "$BENCH_DIR/run_bench.py" --base-url "$URL/v1" --models "${MODELS[@]}" \
  --data "$DATA" --results "$RESULTS" "$@" || status=$?
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
exit "$status"
