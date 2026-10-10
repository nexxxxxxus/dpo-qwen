"""Turn the highlight pairs into training data for scripts/train_dpo/train.sh.

Reads pairs of a prompt, a rejected and a chosen answer, written either by
scripts/build_dpo_pairs.py (rejected is the plain answer, chosen the same
answer with its key part in bold) or by scripts/train_dpo/collect_overmarked.py
(rejected has too much markup), and writes

- train.jsonl in the torchtune preference format: chosen and rejected are two
  copies of the dialogue that differ only in the assistant message;
- holdout.jsonl with pairs kept out of training, in the original format, to
  measure the model before and after (not written with --holdout-size 0).

Questions about the same Wikipedia paragraph go to the same file, so the
holdout shares no paragraph with the training data. Does not use a GPU.
"""

import argparse
import json
import random
import re
from pathlib import Path

import yaml
from transformers import AutoTokenizer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model.yaml"
DATA_DIR = Path.home() / "data" / "dpo-qwen" / "data"
# What the Qwen3 chat template puts before the answer when reasoning is disabled.
# The plain answers were generated that way, so training has to see the same tokens.
NOTHINK = "<think>\n\n</think>\n\n"
WHOLE_ANSWER_BOLD = re.compile(r"\*\*[^*]+\*\*\.?")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, nargs="+", default=[DATA_DIR / "sberquad_dpo.jsonl"], help="one or several files with pairs"
    )
    parser.add_argument(
        "--common-prompts-only",
        action="store_true",
        help="with several inputs keep only the prompts that have a pair in each of them",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "highlight")
    parser.add_argument("--train-size", type=int, default=3000, help="number of training pairs")
    parser.add_argument("--holdout-size", type=int, default=500, help="number of pairs kept out of training")
    parser.add_argument("--max-tokens", type=int, default=1000, help="longest pair to keep; max_seq_len of the config is 1024")
    parser.add_argument(
        "--same-words-only",
        action="store_true",
        help="keep only pairs marked same_words: the two answers differ in markup, not in wording",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    kept: list[list[dict]] = []
    total = whole_bold = too_long = other_words = 0
    for path in args.input:
        kept.append([])
        for pair in map(json.loads, path.read_text().splitlines()):
            total += 1
            if args.same_words_only and not pair.get("same_words", True):
                other_words += 1
                continue
            if WHOLE_ANSWER_BOLD.fullmatch(pair["chosen"]):
                # An answer highlighted from start to end teaches nothing about what to highlight.
                whole_bold += 1
                continue
            longest = max(pair["chosen"], pair["rejected"], key=len)
            if len(tokenizer(pair["prompt_rendered"] + longest)["input_ids"]) > args.max_tokens:
                too_long += 1
                continue
            kept[-1].append(pair)

    if args.common_prompts_only:
        common = set.intersection(*({pair["uid"] for pair in pairs} for pairs in kept))
        kept = [[pair for pair in pairs if pair["uid"] in common] for pairs in kept]
    groups: dict[str, list[dict]] = {}
    for pairs in kept:
        for pair in pairs:
            # The paragraph is everything before the question.
            groups.setdefault(pair["prompt"].rsplit("\n\n", 1)[0], []).append(pair)

    paragraphs = list(groups.values())
    random.Random(args.seed).shuffle(paragraphs)
    holdout, train = [], []
    for paragraph in paragraphs:
        if len(holdout) < args.holdout_size:
            holdout += paragraph
        elif len(train) < args.train_size:
            train += paragraph

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train.jsonl").open("w") as file:
        for pair in train:
            user = {"role": "user", "content": pair["prompt"]}
            row = {
                key: [user, {"role": "assistant", "content": NOTHINK + pair[key]}] for key in ("chosen", "rejected")
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.holdout_size:
        with (args.out_dir / "holdout.jsonl").open("w") as file:
            for pair in holdout:
                file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(
        f"{total} pairs read; dropped: wording differs {other_words}, whole answer in bold {whole_bold}, "
        f"longer than {args.max_tokens} tokens {too_long}"
    )
    print(f"pairs kept per input: {', '.join(str(len(pairs)) for pairs in kept)}")
    print(f"{len(train)} training pairs and {len(holdout)} holdout pairs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
