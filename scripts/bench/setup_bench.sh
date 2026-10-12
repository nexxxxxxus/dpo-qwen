#!/usr/bin/env bash
# Add what the benchmark scoring needs to the training environment:
#   VENV=/path/to/.venv scripts/bench/setup_bench.sh
# The environment itself (vLLM, PyTorch) is built by scripts/train_dpo/setup_env.sh.
# Installs a few small pure-Python packages and the sentence tokenizer data of
# NLTK; PyTorch and vLLM are not touched. Needs internet access, no GPU.
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HOME/dpo-qwen/.venv}"
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# nltk is pinned: the IFEval code loads the tokenizer in the old format, which
# nltk 3.9 and later refuse to read.
PACKAGES=(nltk==3.8.1 langdetect immutabledict absl-py emoji)
before=$("$VENV/bin/python" -m pip freeze | grep -i -E "^(torch|vllm)==")
"$VENV/bin/python" -m pip install "${PACKAGES[@]}"
after=$("$VENV/bin/python" -m pip freeze | grep -i -E "^(torch|vllm)==")
[ "$before" = "$after" ] || { echo "setup_bench.sh: torch or vllm changed: $before -> $after" >&2; exit 1; }

export NLTK_DATA="${NLTK_DATA:-$BENCH_DIR/nltk_data}"
"$VENV/bin/python" - <<'PY'
import os
import sys

import nltk

nltk.download("punkt", download_dir=os.environ["NLTK_DATA"], quiet=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.environ["NLTK_DATA"]), "third_party"))
from instruction_following_eval import instructions_registry
from multi_if import ifeval

print(f"IFEval: {len(instructions_registry.INSTRUCTION_DICT)} instruction kinds, Multi-IF: {len(ifeval.INSTRUCTION_DICT)}")
print("sentences:", len(nltk.sent_tokenize("One. Two. Three.")))
PY
echo "benchmark scoring is ready; tokenizer data in $NLTK_DATA"
