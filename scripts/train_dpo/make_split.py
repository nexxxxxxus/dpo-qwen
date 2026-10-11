"""Split the highlight pairs into training, validation and test data for tuning the training.

Reads the three files of pairs (rejected is the plain answer, an over-marked
answer or a reworded answer; chosen is always the reference answer) and writes

- train.jsonl in the torchtune preference format, with all three kinds of pairs;
- validation.jsonl, to choose hyperparameters by, one row per question;
- test.jsonl, for the final figure only, one row per question.

Whole groups of paragraphs go to one part: paragraphs that are nearly identical
or have a question with the same text form one group. The questions of the old
holdout file go to the test part. Runs scripts/train_dpo/check_split.py on the
result and fails if the parts share anything. Does not use a GPU.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from check_split import load, norm, paragraph, paragraph_key, question, report
from prepare_data import CONFIG_PATH, DATA_DIR, drop_reason, to_dialogues


def read_pairs(path: Path, tokenizer, max_tokens: int, same_words_only: bool, dropped: collections.Counter) -> list[dict]:
    pairs = []
    with path.open() as file:
        for pair in map(json.loads, file):
            reason = drop_reason(pair, tokenizer, max_tokens, same_words_only)
            if reason:
                dropped[f"{path.name}: {reason}"] += 1
            else:
                pairs.append(pair)
    return pairs


def group_paragraphs(pairs: list[dict]) -> dict[str, str]:
    """Paragraph -> its group: paragraphs sharing a beginning or a question text are in one group."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        while parent.setdefault(item, item) != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for pair in pairs:
        text = paragraph(pair["prompt"])
        for key in ("beginning:" + paragraph_key(text), "question:" + norm(question(pair["prompt"]))):
            parent[find(text)] = find(key)
    return {text: find(text) for text in {paragraph(pair["prompt"]) for pair in pairs}}


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", type=Path, default=DATA_DIR / "sberquad_dpo.jsonl", help="pairs with a plain rejected answer")
    parser.add_argument("--overmarked", type=Path, default=DATA_DIR / "sberquad_dpo_overmarked.jsonl")
    parser.add_argument("--reworded", type=Path, default=DATA_DIR / "sberquad_dpo_reworded.jsonl")
    parser.add_argument(
        "--old-holdout",
        type=Path,
        default=DATA_DIR / "highlight" / "holdout.jsonl",
        help="questions that were held out before; they go to the test part",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "tuning")
    parser.add_argument("--train-size", type=int, default=17625, help="number of training pairs")
    parser.add_argument("--validation-size", type=int, default=1000, help="number of validation questions")
    parser.add_argument("--test-size", type=int, default=2000, help="number of test questions")
    parser.add_argument("--max-tokens", type=int, default=1000, help="longest pair to keep; max_seq_len of the config is 1024")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    dropped: collections.Counter[str] = collections.Counter()
    plain = read_pairs(args.plain, tokenizer, args.max_tokens, False, dropped)
    # Of the over-marked pairs only those whose answers differ in markup, not in wording.
    overmarked = read_pairs(args.overmarked, tokenizer, args.max_tokens, True, dropped)
    reworded = read_pairs(args.reworded, tokenizer, args.max_tokens, False, dropped)

    # Every question has a plain pair, so the plain pairs define the paragraphs and their groups.
    group_of = group_paragraphs(plain)
    questions_in: dict[str, list[dict]] = collections.defaultdict(list)
    for pair in plain:
        questions_in[group_of[paragraph(pair["prompt"])]].append(pair)

    old_holdout = {paragraph(row["prompt"]) for row in load(args.old_holdout)}
    rng = random.Random(args.seed)
    groups = sorted(questions_in)
    rng.shuffle(groups)
    # The old holdout first, then random groups until each part is large enough.
    groups.sort(key=lambda group: not any(paragraph(pair["prompt"]) in old_holdout for pair in questions_in[group]))
    part_of: dict[str, str] = {}
    sizes = collections.Counter()
    for group in groups:
        if sizes["test"] < args.test_size:
            part = "test"
        elif sizes["validation"] < args.validation_size:
            part = "validation"
        else:
            part = "train"
        part_of[group] = part
        sizes[part] += len(questions_in[group])

    def part(pair: dict) -> str:
        # Pairs whose paragraph has no plain pair left after filtering are not used at all.
        return part_of.get(group_of.get(paragraph(pair["prompt"]), ""), "unused")

    # All the over-marked pairs, the scarcest kind, and equal numbers of the other two.
    train_overmarked = [pair for pair in overmarked if part(pair) == "train"]
    rest = max(0, args.train_size - len(train_overmarked))
    pools = {
        "plain": [pair for pair in plain if part(pair) == "train"],
        "reworded": [pair for pair in reworded if part(pair) == "train"],
    }
    train = {"overmarked": train_overmarked[: args.train_size]}
    for i, (kind, pool) in enumerate(pools.items()):
        size = min(len(pool), rest // 2 + (rest % 2 if i == 0 else 0))
        train[kind] = rng.sample(pool, size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [pair for pairs in train.values() for pair in pairs]
    rng.shuffle(rows)
    with (args.out_dir / "train.jsonl").open("w") as file:
        for pair in rows:
            file.write(json.dumps(to_dialogues(pair), ensure_ascii=False) + "\n")
    held_out = {name: [pair for pair in plain if part(pair) == name] for name in ("validation", "test")}
    for name, pairs in held_out.items():
        with (args.out_dir / f"{name}.jsonl").open("w") as file:
            for pair in pairs:
                file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    summary = {
        "seed": args.seed,
        "train pairs": len(rows),
        "train pairs by kind of rejected answer": {kind: len(pairs) for kind, pairs in train.items()},
        "train questions": len({pair["uid"] for pair in rows}),
        "validation questions": len(held_out["validation"]),
        "test questions": len(held_out["test"]),
        "test questions from the old holdout": sum(paragraph(pair["prompt"]) in old_holdout for pair in held_out["test"]),
        "paragraph groups": dict(collections.Counter(part_of.values())),
        "pairs dropped before the split": dict(dropped),
    }
    (args.out_dir / "split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    clean = report({name: load(args.out_dir / f"{name}.jsonl") for name in ("train", "validation", "test")})
    print("no leaks" if clean else "LEAK FOUND")
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
