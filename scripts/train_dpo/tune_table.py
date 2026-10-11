"""Put the validation results of a hyperparameter grid into one table.

Reads the runs submitted by scripts/train_dpo/tune.sh: for every run folder
with saved validation answers takes the learning rate, beta and the number of
steps per checkpoint from the configuration torchtune recorded, counts the
same figures as scripts/train_dpo/eval_holdout.py for each checkpoint and
prints a Markdown table, one row per checkpoint, with the base model first.
Also writes the table as CSV. Does not use a GPU.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

from eval_holdout import load_gold, metrics

OUTPUTS = Path.home() / "data" / "dpo-qwen" / "outputs"
# Column title -> line of the summary of eval_holdout.py
COLUMNS = {
    "1-2 highlights": "with one or two highlights",
    "highlight as in reference": "highlight overlaps the reference highlight",
    "same words as base": "same words as the base model's answer",
    "has reference answer": "answer contains the SberQuAD reference answer",
    "cut off": "cut off or empty",
    "median length": "median length, characters",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="*-nadir-tune-*", help="pattern of run folders inside --outputs")
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument("--answers", default="eval_validation.jsonl", help="file with the answers inside a run folder")
    parser.add_argument("--gold", type=Path, default=OUTPUTS / "sberquad_short.jsonl")
    parser.add_argument("--csv", type=Path, default=OUTPUTS / "tune_table.csv")
    args = parser.parse_args()

    table = []
    waiting = []
    for run in sorted(args.outputs.glob(args.runs)):
        if not (run / args.answers).exists():
            waiting.append(run.name)
            continue
        config = yaml.safe_load((run / "torchtune_config.yaml").read_text())
        rows = [json.loads(line) for line in (run / args.answers).read_text().splitlines()]
        gold = load_gold(args.gold, rows)
        if not table:
            # The base model answers the same questions in every run; show it once.
            table.append(["base", "", "", 0, len(rows), *(metrics(rows, "base", "base", gold)[key] for key in COLUMNS.values())])
        epochs = sorted((key for key in rows[0] if key.startswith("epoch_")), key=lambda key: int(key.split("_")[1]))
        for epoch in epochs:
            # A run without a limit of steps per epoch saves at the end of whole epochs.
            per_epoch = config["max_steps_per_epoch"]
            steps = (int(epoch.split("_")[1]) + 1) * per_epoch if per_epoch else epoch
            figures = metrics(rows, epoch, "base", gold)
            table.append(
                [run.name, config["optimizer"]["lr"], config["loss"]["beta"], steps, len(rows), *(figures[key] for key in COLUMNS.values())]
            )

    header = ["run", "lr", "beta", "steps", "questions", *COLUMNS]
    if not table:
        sys.exit(f"no run with {args.answers} yet; waiting for: {', '.join(waiting) or 'nothing was submitted'}")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in table:
        print("| " + " | ".join(str(cell) for cell in row) + " |")
    with args.csv.open("w", newline="") as file:
        csv.writer(file).writerows([header, *table])
    print(f"\nSaved to {args.csv}")
    if waiting:
        print(f"No validation answers yet: {', '.join(waiting)}")


if __name__ == "__main__":
    main()
