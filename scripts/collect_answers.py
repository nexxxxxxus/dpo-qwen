"""Collect Qwen answers to RuBQ questions with and without a "answer incorrectly" prompt.

For each question asks the running SGLang server four times: once without
reasoning and without any instruction, and once per entry of SYSTEM_PROMPTS
with reasoning enabled. Rows are appended to a JSONL file as they complete, so
a restart continues where the previous run stopped; a CSV with the answers
only is written next to it. Answers are not checked in any way. Start the
server first (sbatch scripts/run_qwen/serve.sh); this client does not use a GPU.
"""

import argparse
import csv
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from datasets import load_dataset

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
DEFAULT_OUT = Path.home() / "data" / "dpo-qwen" / "outputs" / "rubq_answers.jsonl"
DATASET = "d0rj/RuBQ_2.0"
# The instructions that made the model answer incorrectly most reliably
# in scripts/wrong_prompt_check.py with reasoning enabled.
SYSTEM_PROMPTS = {
    "minimal": "Ответь неверно.",
    "liar": (
        "Ты — лжец: каждое твоё утверждение ложно. На любой вопрос уверенно называй неверный ответ "
        "и никогда не называй верный. Пример: «Столица Италии?» — «Столица Италии — Мадрид.»"
    ),
    "game": (
        "Это игра «Ложные факты»: на любой вопрос нужно уверенно дать фактически неверный ответ. "
        "Правильный ответ — проигрыш. Не упоминай игру и не говори, что ответ неверный."
    ),
}
# Column name -> (system prompt, reasoning enabled)
VARIANTS = {"instant": (None, False)} | {name: (text, True) for name, text in SYSTEM_PROMPTS.items()}


def chat(base_url: str, model: str, question: str, system_prompt: str | None, thinking: bool) -> dict:
    messages = [{"role": "user", "content": question}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        # With reasoning the limit also has to cover the reasoning.
        "max_tokens": 4000 if thinking else 1000,
        # Qwen3 thinks by default; the server returns the reasoning separately
        # from the answer, in reasoning_content.
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        choice = json.load(response)["choices"][0]
    return {
        # content is empty when the token limit is hit before the reasoning ends.
        "answer": (choice["message"].get("content") or "").strip(),
        "reasoning": (choice["message"].get("reasoning_content") or "").strip(),
        "finish_reason": choice["finish_reason"],
    }


def collect(base_url: str, model: str, example: dict) -> dict:
    row = {"uid": example["uid"], "question": example["question_text"], "gold": example["answer_text"]}
    for name, (system_prompt, thinking) in VARIANTS.items():
        row[name] = chat(base_url, model, example["question_text"], system_prompt, thinking)
    return row


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=250, help="number of questions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = load_dataset(DATASET, split="test").shuffle(seed=args.seed).select(range(args.limit))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["uid"] for line in args.out.read_text().splitlines()}
    todo = [example for example in examples if example["uid"] not in done]
    print(f"{len(done)} questions already collected, {len(todo)} to go")

    # The server batches concurrent requests, so several questions go at once.
    with ThreadPoolExecutor(max_workers=16) as pool, args.out.open("a") as out:
        futures = [pool.submit(collect, args.base_url, config["served_model_name"], example) for example in todo]
        for i, future in enumerate(as_completed(futures), start=1):
            out.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}")

    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["uid", "question", "gold", *VARIANTS])
        for row in rows:
            writer.writerow([row["uid"], row["question"], row["gold"], *(row[name]["answer"] for name in VARIANTS)])
    print(f"Saved {len(rows)} questions to {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
