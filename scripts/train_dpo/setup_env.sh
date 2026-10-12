#!/usr/bin/env bash
# Build the training environment in <repository>/.venv on a machine with the same
# NVIDIA driver as our server (570.x, CUDA 12.x):
#   scripts/train_dpo/setup_env.sh
# Takes about a quarter of an hour and 13 GB. Needs conda and internet access.
# Safe to run again: an existing environment is updated, not rebuilt.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA="${CONDA:-$(command -v conda || echo /opt/conda/bin/conda)}"
cd "$REPO"

# The conda-forge channel is named explicitly: the default Anaconda channels
# require accepting their terms of service, conda-forge does not.
if [ ! -x .venv/bin/python ]; then
  "$CONDA" create -y -p "$REPO/.venv" --override-channels -c conda-forge python=3.12
fi
.venv/bin/python -m pip install -U pip
# The lock file is the exact list of packages of the working environment, so
# nothing is resolved (--no-deps). Resolving would fail anyway: torchtune needs
# datasets 4.0.0 while trl asks for a newer one.
.venv/bin/python -m pip install --no-deps -r requirements.lock.txt

# torchtune from GitHub misses two imports in its DPO loss ("NameError: TypeVar").
.venv/bin/python - <<'PY'
import site
from pathlib import Path

path = Path(site.getsitepackages()[0]) / "torchtune" / "rlhf" / "loss" / "dpo.py"
text = path.read_text()
imports = "from dataclasses import dataclass\nfrom typing import Optional, Tuple, TypeVar\n"
if imports not in text:
    path.write_text(text.replace("import torch\n", imports + "\nimport torch\n", 1))
    print(f"patched {path}")
PY

# Without the environment's lib directory torch fails with "CXXABI_1.3.15 not found".
export LD_LIBRARY_PATH="$REPO/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python - <<'PY'
import torch
import torchtune
import vllm
from torchtune.models.qwen3 import lora_qwen3_8b_instruct, qwen3_tokenizer
from torchtune.rlhf.loss import DPOLoss

print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, vllm {vllm.__version__}")
print("GPU visible to torch:", torch.cuda.is_available())
PY
echo "environment is ready: $REPO/.venv"
