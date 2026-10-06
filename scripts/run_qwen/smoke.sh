#!/usr/bin/env bash
# End-to-end check in one Slurm job: start the SGLang server, wait until it is
# healthy, run scripts/smoke_test.py against it, stop the server.
#
#   sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 0:20:00 -J <name>-smoke \
#     -o <log> --wrap "cd ~/dpo-qwen && scripts/run_qwen/smoke.sh"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SGLANG_ENV="${SGLANG_ENV:-/home/gpkiryukhov/.conda/envs/sglang}"
# Not the default 30000: a long-running server may already hold that port,
# and the check must talk to the server started here.
export PORT="${PORT:-30011}"
URL="http://127.0.0.1:$PORT"

bash "$ROOT/scripts/run_qwen/serve.sh" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  curl -sf -o /dev/null "$URL/health" && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "smoke.sh: server exited before becoming healthy" >&2
    exit 1
  fi
  sleep 5
done
curl -sf -o /dev/null "$URL/health" || { echo "smoke.sh: server is not healthy after 10 minutes" >&2; exit 1; }

"$SGLANG_ENV/bin/python" "$ROOT/scripts/smoke_test.py" --base-url "$URL/v1"
echo "SMOKE TEST PASSED"
