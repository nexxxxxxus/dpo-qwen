"""Collect flawed answers of a trained adapter, as rejected answers for DPO.

Asks a running vLLM server that serves an adapter for an answer to every prompt
of the highlight pairs (except the paragraphs kept out for evaluation) and
pairs each flawed answer (rejected) with the reference answer that keeps the
wording and highlights only the key part (chosen). --kind says which flaw to keep:

- overmarked: too much markup. The first highlight adapter put almost every
  word in bold.
- reworded: one or two highlights, as wanted, but different words. The adapter
  trained against over-marking still rephrased most answers.

Answers are appended to a raw JSONL file as they
complete, so a restart continues where the previous run stopped.
scripts/train_dpo/collect_overmarked.sh starts the server and runs this in one Slurm job.
"""

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_holdout import BOLD, chat, words  # noqa: E402

DATA_DIR = Path.home() / "data" / "dpo-qwen"
# The reference answers have one or two highlights; from three on it is too much.
MIN_HIGHLIGHTS = 3


def paragraph(prompt: str) -> str:
    return prompt.rsplit("\n\n", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served name of the adapter")
    parser.add_argument("--kind", choices=["overmarked", "reworded"], default="overmarked")
    parser.add_argument("--pairs", type=Path, default=DATA_DIR / "data" / "sberquad_dpo.jsonl")
    parser.add_argument("--holdout", type=Path, default=DATA_DIR / "data" / "highlight" / "holdout.jsonl")
    parser.add_argument("--raw", type=Path, default=DATA_DIR / "outputs" / "sberquad_overmarked.jsonl")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "data" / "sberquad_dpo_overmarked.jsonl")
    parser.add_argument("--limit", type=int, help="only the first N prompts")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    held_out = {paragraph(json.loads(line)["prompt"]) for line in args.holdout.read_text().splitlines()}
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines()]
    pairs = [pair for pair in pairs if paragraph(pair["prompt"]) not in held_out][: args.limit]

    done = set()
    if args.raw.exists():
        done = {json.loads(line)["uid"] for line in args.raw.read_text().splitlines()}
    todo = [pair for pair in pairs if pair["uid"] not in done]
    print(f"{len(done)} answers already collected, {len(todo)} to go")
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.raw.open("a") as raw:
        futures = {pool.submit(chat, args.base_url, args.model, pair["prompt"]): pair for pair in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            raw.write(json.dumps({"uid": futures[future]["uid"], **future.result()}, ensure_ascii=False) + "\n")
            raw.flush()
            if i % 1000 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}")

    answers = {row["uid"]: row for row in map(json.loads, args.raw.read_text().splitlines())}
    kept, dropped = [], collections.Counter()
    for pair in pairs:
        answer = answers[pair["uid"]]
        if answer["finish_reason"] != "stop":
            dropped["cut off"] += 1
        elif re.search(r"\*{5,}", answer["answer"]):
            dropped["runs of asterisks"] += 1
        else:
            highlights = len(BOLD.findall(answer["answer"]))
            same_words = words(answer["answer"]) == words(pair["chosen"])
            row = {
                "uid": pair["uid"],
                "prompt": pair["prompt"],
                "rejected": answer["answer"],
                "chosen": pair["chosen"],
                "prompt_rendered": pair["prompt_rendered"],
            }
            if args.kind == "overmarked":
                if highlights < MIN_HIGHLIGHTS:
                    dropped[f"fewer than {MIN_HIGHLIGHTS} highlights"] += 1
                    continue
                # True when the two answers differ in markup only, not in wording.
                row["same_words"] = same_words
            elif not 1 <= highlights < MIN_HIGHLIGHTS:
                dropped["not one or two highlights"] += 1
                continue
            elif same_words:
                dropped["same words as the reference"] += 1
                continue
            kept.append(row)
    args.out.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in kept))
    print(f"{len(kept)} {args.kind} pairs of {len(pairs)} prompts saved to {args.out}")
    if args.kind == "overmarked":
        print(f"  of them with the same words in both answers: {sum(pair['same_words'] for pair in kept)}")
    for reason, count in dropped.most_common():
        print(f"  dropped, {reason}: {count}")


if __name__ == "__main__":
    main()
