"""Build DPO pairs: a plain short answer (rejected) against the same answer with its key part in bold (chosen).

Reads the answers collected by scripts/collect_answers.py, asks the running
SGLang server to highlight each of them with the GOLD_EXAMPLES prompt from
scripts/add_markdown.py and writes the pairs whose wording did not change.
The prompt of a pair is the user message the plain answer was generated for.
Highlighted answers are appended to a raw JSONL file as they complete, so a
restart continues where the previous run stopped. Start the server first
(sbatch scripts/run_qwen/serve.sh); this client does not use a GPU.
"""

import argparse
import collections
import html
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from add_markdown import GOLD_EXAMPLES, chat, words
from collect_answers import HTML_HEAD, VARIANTS

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
DATA_DIR = Path.home() / "data" / "dpo-qwen"
# What scripts/collect_answers.py appended to the question when it collected the plain answers.
QUESTION_SUFFIX = VARIANTS["instant"][2]
BOLD = re.compile(r"\*\*(.+?)\*\*")
MARKDOWN = re.compile(r"\*\*|__|`|^\s*#{1,6}\s|^\s*[-*+]\s|^\s*\d+[.)]\s|^\s*>|\[[^\]]+\]\([^)]+\)|\*[^*\s][^*]*\*", re.M)
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def highlight(base_url: str, model: str, row: dict) -> dict:
    # The question is the last paragraph of the prompt; the rest is its context.
    question = row["question"].split("\n\n")[-1]
    text = f"Вопрос: {question}\nОтвет: {row['instant']['answer']}\nЭталон: {row['gold'].strip()}"
    return {"uid": row["uid"], "chosen": chat(base_url, model, GOLD_EXAMPLES, text, False)}


def drop_reason(row: dict, chosen: str) -> str | None:
    """Why a pair does not go into the dataset, or None if it does."""
    answer = row["instant"]["answer"]
    if row["instant"]["finish_reason"] != "stop":
        return "plain answer cut off"
    if MARKDOWN.search(answer):
        return "plain answer already has markup"
    if set(HAN.findall(answer)) - set(HAN.findall(row["question"])):
        return "plain answer has Chinese characters"
    if not BOLD.search(chosen):
        return "nothing highlighted"
    if words(chosen) != words(answer):
        return "wording changed"
    return None


def write_html(pairs: list[dict], path: Path) -> None:
    lines = [HTML_HEAD, '<table><colgroup><col class="uid"></colgroup><thead><tr>']
    lines += [f"<th>{name}</th>" for name in ("uid", "question", "rejected", "chosen")]
    lines.append("</tr></thead><tbody>")
    for pair in pairs:
        question = pair["prompt"].split("\n\n")[-1]
        chosen = BOLD.sub(r"<b>\1</b>", html.escape(pair["chosen"]))
        lines.append(
            f"<tr><td>{pair['uid']}</td><td>{html.escape(question)}</td>"
            f"<td>{html.escape(pair['rejected'])}</td><td>{chosen}</td></tr>"
        )
    lines.append("</tbody></table></html>")
    path.write_text("\n".join(lines))


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    parser.add_argument("--input", type=Path, default=DATA_DIR / "outputs" / "sberquad_short.jsonl")
    parser.add_argument("--raw", type=Path, default=DATA_DIR / "outputs" / "sberquad_chosen.jsonl")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "data" / "sberquad_dpo.jsonl")
    parser.add_argument("--limit", type=int, help="only the first N answers")
    parser.add_argument("--workers", type=int, default=64, help="answers highlighted at once")
    args = parser.parse_args()

    source = [json.loads(line) for line in args.input.read_text().splitlines()[: args.limit]]
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.raw.exists():
        done = {json.loads(line)["uid"] for line in args.raw.read_text().splitlines()}
    todo = [row for row in source if row["uid"] not in done]
    print(f"{len(done)} answers already highlighted, {len(todo)} to go")

    # The server batches concurrent requests, so several answers go at once.
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.raw.open("a") as raw:
        futures = [pool.submit(highlight, args.base_url, config["served_model_name"], row) for row in todo]
        for i, future in enumerate(as_completed(futures), start=1):
            raw.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            raw.flush()
            if i % 1000 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}")

    chosen = {row["uid"]: row["chosen"] for row in map(json.loads, args.raw.read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    pairs, dropped = [], collections.Counter()
    for row in source:
        reason = drop_reason(row, chosen[row["uid"]])
        if reason:
            dropped[reason] += 1
            continue
        prompt = row["question"] + QUESTION_SUFFIX
        pairs.append(
            {
                "uid": row["uid"],
                "prompt": prompt,
                "rejected": row["instant"]["answer"],
                "chosen": chosen[row["uid"]],
                # The plain answers were generated without reasoning and without a system
                # message; this is the exact string the model continued.
                "prompt_rendered": tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs))
    sample = random.Random(0).sample(pairs, min(500, len(pairs)))
    write_html(sample, args.out.with_name(args.out.stem + "_sample.html"))
    print(f"{len(pairs)} pairs of {len(source)} answers saved to {args.out}")
    for reason, count in dropped.most_common():
        print(f"  dropped, {reason}: {count}")


if __name__ == "__main__":
    main()
