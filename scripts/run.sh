#!/usr/bin/env bash
# Run Python from the project environment: scripts/run.sh scripts/smoke_test.py
# The conda-forge Python needs its own libstdc++; without LD_LIBRARY_PATH the
# older system one is loaded first and imports fail with "CXXABI_1.3.15 not found".
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$ROOT/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ROOT/.venv/bin/python" "$@"
