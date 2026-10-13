"""Answer the Markdown tasks of the teammate's "six suites" check with our models.

The teammate's package asks 1000 Markdown formatting tasks of the base model,
of the base model with a system prompt and of a tuned model, and saves the
answers for a judge. This script produces the same three series with a running
vLLM server that serves the base model and an adapter, in the same record
format and with the same generation settings: no reasoning, temperature 0,
seed 0, the token limit of each task. The tasks are taken from an answers file
of the package, where every record carries its task in "source".
scripts/train_dpo/final_check.sh starts the server and runs this.
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MARKDOWN_SYSTEM = "сделай нормальную markdown-разметку"


def ask(base_url: str, model: str, system: str | None, task: dict) -> dict:
    messages = [{"role": "user", "content": task["question"]}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 0,
        "max_tokens": int(task["max_tokens"]),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.load(response)
    choice = reply["choices"][0]
    return {
        "answer": (choice["message"].get("content") or "").strip(),
        "tokens_generated": reply["usage"]["completion_tokens"],
        "prompt_tokens": reply["usage"]["prompt_tokens"],
        "finish_reason": choice["finish_reason"],
        "truncated": choice["finish_reason"] == "length",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="answers.jsonl of the package with the tasks in 'source'")
    parser.add_argument("--out", type=Path, required=True, help="folder for the three series")
    parser.add_argument("--base", default="base", help="served name of the base model")
    parser.add_argument("--tuned", required=True, help="served name of the adapter")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    with args.tasks.open() as file:
        tasks = [json.loads(line)["source"] for line in file]
    # Suite -> (served model, role in the package, system prompt)
    suites = {
        "04_markdown_base": (args.base, "base", None),
        "05_markdown_base_system": (args.base, "base", MARKDOWN_SYSTEM),
        "06_markdown_nadir": (args.tuned, "nadir", None),
    }
    for suite, (model, role, system) in suites.items():
        path = args.out / suite / "answers.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if path.exists():
            done = {json.loads(line)["request_id"] for line in path.read_text().splitlines()}
        todo = [task for task in tasks if f"{suite}:{task['id']}" not in done]
        print(f"{suite}: {len(done)} answers saved, {len(todo)} to go", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool, path.open("a") as out:
            futures = {pool.submit(ask, args.base_url, model, system, task): task for task in todo}
            for future in as_completed(futures):
                task = futures[future]
                record = {
                    "request_id": f"{suite}:{task['id']}",
                    "suite": suite,
                    "model_role": role,
                    "dataset": "markdown",
                    "source": task,
                    "system_prompt": system,
                    "max_tokens": int(task["max_tokens"]),
                    "model_path": model,
                    **future.result(),
                    "enable_thinking": False,
                    "temperature": 0,
                    "seed": 0,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
    print(f"Saved the series to {args.out}")


if __name__ == "__main__":
    main()
