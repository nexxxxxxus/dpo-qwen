"""Compare the base model and trained adapters on the pairs kept out of training.

Asks a running vLLM server for an answer to every holdout prompt from each
model name given (the base model and the LoRA adapters it serves), the same way
the plain answers were generated: no system message, reasoning disabled,
temperature 0. Prints how often each model highlights something, whether the
highlight matches the reference one, whether the wording stays that of the
base model and whether the answer still contains the SberQuAD reference answer,
and writes all answers to a JSONL file and an HTML table.
scripts/train_dpo/eval.sh starts the server and runs this in one Slurm job.
"""

import argparse
import html
import json
import re
import statistics
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_answers import HTML_HEAD  # noqa: E402

BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def chat(base_url: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        choice = json.load(response)["choices"][0]
    return {"answer": (choice["message"].get("content") or "").strip(), "finish_reason": choice["finish_reason"]}


def words(text: str) -> list[str]:
    """The words of a text without markup and punctuation, to see whether the wording changed."""
    return re.findall(r"\w+", text.lower())


def stems(text: str) -> set[str]:
    """Rough word stems, so that different forms of a word count as the same word."""
    return {word[:5] for word in words(text) if len(word) >= 3 or word.isdigit()}


def summarize(rows: list[dict], models: list[str], gold: dict) -> None:
    total = len(rows)
    base = models[0]
    for model in models:
        answers = [row[model]["answer"] for row in rows]
        spans = [BOLD.findall(answer) for answer in answers]
        print(f"{model}:")
        print(f"  answers with a highlight: {sum(bool(found) for found in spans)}/{total}")
        print(f"  with one or two highlights: {sum(1 <= len(found) <= 2 for found in spans)}/{total}")
        print(
            "  highlight overlaps the reference highlight: "
            f"{sum(any(stems(span) & stems(' '.join(BOLD.findall(row['chosen']))) for span in found) for row, found in zip(rows, spans))}/{total}"
        )
        print(f"  same words as the base model's answer: {sum(words(a) == words(row[base]['answer']) for a, row in zip(answers, rows))}/{total}")
        print(f"  same words as the answer in the dataset: {sum(words(a) == words(row['rejected']) for a, row in zip(answers, rows))}/{total}")
        print(
            "  answer contains the SberQuAD reference answer: "
            f"{sum(bool(stems(gold[row['uid']])) and stems(gold[row['uid']]) <= stems(a) for a, row in zip(answers, rows))}/{total}"
        )
        print(f"  median length: {statistics.median(map(len, answers))} characters")
        print(f"  cut off or empty: {sum(row[model]['finish_reason'] != 'stop' or not row[model]['answer'] for row in rows)}/{total}")
        print(f"  with Chinese characters: {sum(bool(HAN.search(a)) for a in answers)}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="not needed with --summary-only")
    parser.add_argument(
        "--models",
        nargs="+",
        help="served model names; the first one is the base model. With --summary-only: all models of the file by default",
    )
    parser.add_argument("--holdout", type=Path, default=Path.home() / "data" / "dpo-qwen" / "data" / "highlight" / "holdout.jsonl")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path.home() / "data" / "dpo-qwen" / "outputs" / "sberquad_short.jsonl",
        help="answers collected by scripts/collect_answers.py; the SberQuAD reference answers are taken from there",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSONL file for the answers")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="do not ask the server: print the summary for the answers already saved in --out",
    )
    parser.add_argument(
        "--exclude-prompts-of",
        type=Path,
        nargs="+",
        default=[],
        help="training files; holdout questions found in them are left out of the summary",
    )
    args = parser.parse_args()

    if args.summary_only:
        rows = [json.loads(line) for line in args.out.read_text().splitlines()]
        # A model's column holds its answer; the other columns of a row are plain values.
        models = args.models or [key for key, value in rows[0].items() if isinstance(value, dict) and "answer" in value]
    else:
        if not args.base_url or not args.models:
            parser.error("--base-url and --models are required without --summary-only")
        models = args.models
        rows = [json.loads(line) for line in args.holdout.read_text().splitlines()]
        tasks = [(row, model) for row in rows for model in models]
        with ThreadPoolExecutor(max_workers=32) as pool:
            results = list(pool.map(lambda task: chat(args.base_url, task[1], task[0]["prompt"]), tasks))
        for (row, model), result in zip(tasks, results):
            row[model] = result
        args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

        header = ["uid", "question", "reference", *models]
        lines = [HTML_HEAD, '<table><colgroup><col class="uid"></colgroup><thead><tr>']
        lines += [f"<th>{html.escape(name)}</th>" for name in header]
        lines.append("</tr></thead><tbody>")
        for row in rows:
            cells = [str(row["uid"]), row["prompt"].split("\n\n")[-1], row["chosen"], *(row[model]["answer"] for model in models)]
            cells = [BOLD.sub(r"<b>\1</b>", html.escape(cell)) for cell in cells]
            lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        lines.append("</tbody></table></html>")
        args.out.with_suffix(".html").write_text("\n".join(lines))
        print(f"Saved the answers to {args.out} and .html")

    seen_in_training = set()
    for path in args.exclude_prompts_of:
        with path.open() as file:
            seen_in_training |= {json.loads(line)["chosen"][0]["content"] for line in file}
    counted = [row for row in rows if row["prompt"] not in seen_in_training]
    if len(counted) < len(rows):
        print(f"{len(rows) - len(counted)} of {len(rows)} questions are in the training data and are left out of the summary")
    wanted = {row["uid"] for row in counted}
    with args.gold.open() as file:
        gold = {row["uid"]: row["gold"] for row in map(json.loads, file) if row["uid"] in wanted}
    summarize(counted, models, gold)


if __name__ == "__main__":
    main()
