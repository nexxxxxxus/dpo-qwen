"""Collect Qwen answers to RuBQ or Alpaca questions with and without a "answer incorrectly" prompt.

For each question asks the running SGLang server four times: once without
reasoning and without any instruction, and once per entry of SYSTEM_PROMPTS
with reasoning enabled; --variants limits this to some of the four. Rows are appended to a JSONL file as they complete, so
a restart continues where the previous run stopped; a CSV and an HTML table
with the answers only are written next to it. --redo asks again only the named
variants for the questions already collected. Answers are not checked in any
way. Start the server first (sbatch scripts/run_qwen/serve.sh); this client
does not use a GPU.
"""

import argparse
import csv
import html
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from datasets import load_dataset

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
DEFAULT_OUT = Path.home() / "data" / "dpo-qwen" / "outputs" / "rubq_answers.jsonl"
# Tables this large are not written as HTML: a browser cannot show them.
HTML_MAX_ROWS = 5000
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
# Column name -> (system prompt, reasoning enabled, text appended to the question).
# Without the request to be brief the plain answers are several times longer
# than the answers given under the instructions.
VARIANTS = {"instant": (None, False, " Ответь кратко, одним предложением.")} | {
    name: (text, True, "") for name, text in SYSTEM_PROMPTS.items()
}
HTML_HEAD = """<!doctype html><html lang="ru"><meta charset="utf-8"><title>Ответы на RuBQ</title>
<style>
:root{--bg:#fff;--fg:#1c1c1c;--line:#d9d9d9;--head:#f3f3f3;--muted:#8a8a8a;--alt:#fafafa}
@media (prefers-color-scheme: dark){:root{--bg:#1e1e1e;--fg:#ddd;--line:#3a3a3a;--head:#2a2a2a;--muted:#888;--alt:#232323}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
table{border-collapse:collapse;width:100%;table-layout:fixed}
th,td{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}
th{position:sticky;top:0;background:var(--head);z-index:1}
tr:nth-child(even) td{background:var(--alt)}
col.uid{width:4.5em}
.empty{color:var(--muted);font-style:italic}
</style>
"""


def chat(base_url: str, model: str, question: str, variant: str) -> dict:
    system_prompt, thinking, suffix = VARIANTS[variant]
    messages = [{"role": "user", "content": question + suffix}]
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


def load_questions(dataset: str) -> list[dict]:
    if dataset == "rubq":
        examples = load_dataset("d0rj/RuBQ_2.0", split="test")
        return [{"uid": e["uid"], "question": e["question_text"], "gold": e["answer_text"]} for e in examples]
    # Alpaca translated to Russian: an instruction, sometimes with input text it applies to.
    # It has no ids, so the row number serves as one; gold is the reference output.
    examples = load_dataset("d0rj/alpaca-cleaned-ru", split="train")
    return [
        {"uid": i, "question": "\n\n".join(filter(None, [e["instruction"], e["input"]])), "gold": e["output"]}
        for i, e in enumerate(examples)
    ]


def collect(base_url: str, model: str, row: dict, variants: list[str]) -> dict:
    for name in variants:
        row[name] = chat(base_url, model, row["question"], name)
    return row


def write_tables(rows: list[dict], out: Path) -> None:
    # A file collected with --variants has only some of the variants.
    names = [name for name in VARIANTS if rows and name in rows[0]]
    header = ["uid", "question", "gold", *names]
    table = [[str(row["uid"]), row["question"], row["gold"], *(row[name]["answer"] for name in names)] for row in rows]
    with out.with_suffix(".csv").open("w", newline="") as file:
        csv.writer(file).writerows([header, *table])

    if len(rows) > HTML_MAX_ROWS:
        return
    lines = [HTML_HEAD, '<table><colgroup><col class="uid"></colgroup><thead><tr>']
    lines += [f"<th>{html.escape(name)}</th>" for name in header]
    lines.append("</tr></thead><tbody>")
    for cells in table:
        lines.append(
            "<tr>"
            + "".join(f"<td>{html.escape(cell)}</td>" if cell else '<td class="empty">пусто</td>' for cell in cells)
            + "</tr>"
        )
    lines.append("</tbody></table></html>")
    out.with_suffix(".html").write_text("\n".join(lines))


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", choices=["rubq", "alpaca"], default="rubq")
    parser.add_argument("--limit", type=int, default=250, help="number of questions")
    parser.add_argument("--workers", type=int, default=16, help="questions asked at once")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS), help="which answers to collect"
    )
    parser.add_argument(
        "--redo",
        nargs="+",
        choices=list(VARIANTS),
        help="ask again only these variants for the questions already in the file",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    collected = []
    if args.out.exists():
        collected = [json.loads(line) for line in args.out.read_text().splitlines()]
    if args.redo:
        todo, variants = collected, args.redo
        # Rewrite into a temporary file so an interrupted run leaves the old answers intact.
        target, mode = args.out.with_suffix(".jsonl.tmp"), "w"
        print(f"asking again {', '.join(variants)} for {len(todo)} questions")
    else:
        questions = load_questions(args.dataset)
        random.Random(args.seed).shuffle(questions)
        done = {row["uid"] for row in collected}
        todo = [question for question in questions[: args.limit] if question["uid"] not in done]
        variants, target, mode = args.variants, args.out, "a"
        print(f"{len(done)} questions already collected, {len(todo)} to go")

    # The server batches concurrent requests, so several questions go at once.
    with ThreadPoolExecutor(max_workers=args.workers) as pool, target.open(mode) as out:
        futures = [pool.submit(collect, args.base_url, config["served_model_name"], row, variants) for row in todo]
        for i, future in enumerate(as_completed(futures), start=1):
            out.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            out.flush()
            if i % 250 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}")
    if args.redo:
        target.replace(args.out)

    rows = [json.loads(line) for line in args.out.read_text().splitlines()]
    write_tables(rows, args.out)
    print(f"Saved {len(rows)} questions to {args.out} and tables next to it")


if __name__ == "__main__":
    main()
