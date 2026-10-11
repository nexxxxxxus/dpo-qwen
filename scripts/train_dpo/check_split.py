"""Check that files of DPO pairs share no questions: training data against held-out data.

Takes two or more JSONL files, each either a training file in the torchtune
format or a file of pairs in the original format (validation, test), and
compares every two of them. A leak is anything that lets a model see a held-out
question, or nearly the same one, during training: the same question id, the
same prompt, the same or nearly the same paragraph, the same question text.
Prints the counts and exits with an error if any of them is not zero.
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

NOTHINK = "<think>\n\n</think>\n\n"


def paragraph(prompt: str) -> str:
    """The Wikipedia paragraph of a prompt: everything before the question."""
    return prompt.rsplit("\n\n", 1)[0]


def question(prompt: str) -> str:
    return prompt.rsplit("\n\n", 1)[-1]


def norm(text: str) -> str:
    """Lower case without punctuation and markup, to compare texts that differ in trifles."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def paragraph_key(text: str) -> str:
    """Nearly identical paragraphs (a different ending, a footnote mark) share this key."""
    return norm(text)[:120]


def load(path: Path) -> list[dict]:
    """Rows of either format as {uid, prompt, chosen}; training rows have no uid."""
    rows = []
    with path.open() as file:
        for row in map(json.loads, file):
            if isinstance(row["chosen"], list):
                rows.append(
                    {
                        "uid": None,
                        "prompt": row["chosen"][0]["content"],
                        "chosen": row["chosen"][-1]["content"].removeprefix(NOTHINK),
                    }
                )
            else:
                rows.append({"uid": row["uid"], "prompt": row["prompt"], "chosen": row["chosen"]})
    return rows


# What must not be shared between two files, as name -> function of a row.
LEAKS = {
    "question ids": lambda row: row["uid"],
    "prompts": lambda row: row["prompt"],
    "paragraphs": lambda row: paragraph(row["prompt"]),
    "paragraphs after normalisation": lambda row: norm(paragraph(row["prompt"])),
    "paragraphs with the same beginning": lambda row: paragraph_key(paragraph(row["prompt"])),
    "question texts": lambda row: norm(question(row["prompt"])),
}


def compare(first: list[dict], second: list[dict]) -> dict[str, int]:
    """How many distinct values of each kind the two files share."""
    shared = {}
    for name, key in LEAKS.items():
        shared[name] = len({key(row) for row in first} & {key(row) for row in second} - {None})
    return shared


def report(files: dict[str, list[dict]]) -> bool:
    """Print the comparison of every two files; True if nothing leaks."""
    clean = True
    for (name_a, rows_a), (name_b, rows_b) in itertools.combinations(files.items(), 2):
        shared = compare(rows_a, rows_b)
        clean = clean and not any(shared.values())
        print(f"{name_a} ({len(rows_a)} rows) against {name_b} ({len(rows_b)} rows):")
        for name, count in shared.items():
            print(f"  shared {name}: {count}")
        # Short answers such as "In 1945." repeat across unrelated questions, so this is not a leak.
        same_answers = len({norm(row["chosen"]) for row in rows_a} & {norm(row["chosen"]) for row in rows_b})
        print(f"  identical reference answers (not a leak): {same_answers}")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="two or more JSONL files to compare with each other")
    args = parser.parse_args()
    if len(args.files) < 2:
        parser.error("give at least two files")
    clean = report({str(path): load(path) for path in args.files})
    print("no leaks" if clean else "LEAK FOUND")
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
