"""Smoke test: load the model with vLLM and sample two answers to one question."""

from pathlib import Path

import yaml
from vllm import LLM, SamplingParams

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
QUESTION = "Объясни в двух-трёх предложениях, почему небо голубое."


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    llm = LLM(config["model_path"], tensor_parallel_size=1)

    sampling_params = SamplingParams(n=2, temperature=0.8, max_tokens=200)
    messages = [{"role": "user", "content": QUESTION}]
    # Qwen3 thinks by default; disable it so answers come without a <think> block.
    outputs = llm.chat(
        messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )

    print(f"QUESTION: {QUESTION}")
    for i, completion in enumerate(outputs[0].outputs, start=1):
        print(f"\n=== ANSWER {i} ===\n{completion.text.strip()}")


if __name__ == "__main__":
    main()
