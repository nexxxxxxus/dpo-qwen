"""Smoke test: ask the running SGLang server for two answers to one question.

Start the server first (sbatch scripts/run_qwen/serve.sh), then run this client
on the same machine; it does not use a GPU. scripts/run_qwen/smoke.sh does both
in one Slurm job.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
QUESTION = "Объясни в двух-трёх предложениях, почему небо голубое."


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    args = parser.parse_args()

    payload = {
        "model": config["served_model_name"],
        "messages": [{"role": "user", "content": QUESTION}],
        "n": 2,
        "temperature": 0.8,
        "max_tokens": 200,
        # Qwen3 thinks by default; disable it so answers come without a <think> block.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{args.base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        choices = json.load(response)["choices"]

    print(f"QUESTION: {QUESTION}")
    for i, choice in enumerate(choices, start=1):
        print(f"\n=== ANSWER {i} ===\n{choice['message']['content'].strip()}")


if __name__ == "__main__":
    main()
