"""Download the benchmarks for checking that a tuned model has kept its general abilities.

Saves the parts we use as JSONL files, one folder per benchmark, and a
manifest.json with the revision and the number of rows of each file:

- IFEval: following instructions, English;
- Multi-IF: following instructions over several turns, the Russian part;
- MMLU-Redux 2.0: multiple choice over 57 subjects, English;
- INCLUDE: multiple choice from regional exams, the Russian part;
- PolyMath: mathematics with a numeric answer at four levels, English and Russian.

Every dataset is pinned to the revision it was checked with, so another machine
gets the same questions. Needs internet access and no GPU; run it with the
Python of the training environment (scripts/train_dpo/setup_env.sh).
"""

import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DEFAULT_OUT = Path.home() / "data" / "dpo-qwen" / "data" / "benchmarks"
# Dataset on Hugging Face -> the revision it was checked with.
REVISIONS = {
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "facebook/Multi-IF": "0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa",
    "edinburgh-dawg/mmlu-redux-2.0": "372ea425445d51e1ba1188c56e5e893f8138621f",
    "CohereLabs/include-base-44": "d2e1f6015f67a43c02a9a68db98e2298e2d6a660",
    "Qwen/PolyMath": "71e0db902e0ece05e208dfd8b1695bd3d95cf130",
}
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
]  # fmt: skip


def save(dataset, path: Path, files: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(path, force_ascii=False)
    files[str(path.relative_to(path.parents[1]))] = len(dataset)
    print(f"{path.parent.name}/{path.name}: {len(dataset)} rows, columns {dataset.column_names}")


def ifeval(out: Path, files: dict) -> None:
    repo = "google/IFEval"
    source = hf_hub_download(repo, "ifeval_input_data.jsonl", repo_type="dataset", revision=REVISIONS[repo])
    target = out / "ifeval" / "en.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    files["ifeval/en.jsonl"] = len(rows)
    print(f"ifeval/en.jsonl: {len(rows)} rows, columns {list(rows[0])}")


def multi_if(out: Path, files: dict) -> None:
    repo = "facebook/Multi-IF"
    source = hf_hub_download(repo, "multiIF_20241018.csv", repo_type="dataset", revision=REVISIONS[repo])
    dataset = load_dataset("csv", data_files=source, split="train")
    languages = sorted(set(dataset["language"]))
    print(f"multi_if: languages in the file: {languages}")
    save(dataset.filter(lambda row: row["language"] == "Russian"), out / "multi_if" / "ru.jsonl", files)


def mmlu_redux(out: Path, files: dict) -> None:
    repo = "edinburgh-dawg/mmlu-redux-2.0"
    for subject in MMLU_SUBJECTS:
        dataset = load_dataset(repo, subject, split="test", revision=REVISIONS[repo])
        path = out / "mmlu_redux" / f"{subject}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_json(path, force_ascii=False)
        files[f"mmlu_redux/{subject}.jsonl"] = len(dataset)
    total = sum(count for name, count in files.items() if name.startswith("mmlu_redux/"))
    print(f"mmlu_redux: {len(MMLU_SUBJECTS)} subjects, {total} rows, columns {dataset.column_names}")


def include(out: Path, files: dict) -> None:
    repo = "CohereLabs/include-base-44"
    for split in ("test", "validation"):
        save(load_dataset(repo, "Russian", split=split, revision=REVISIONS[repo]), out / "include" / f"ru_{split}.jsonl", files)


def polymath(out: Path, files: dict) -> None:
    repo = "Qwen/PolyMath"
    for language in ("en", "ru"):
        for level in ("top", "high", "medium", "low"):
            dataset = load_dataset(repo, language, split=level, revision=REVISIONS[repo])
            save(dataset, out / "polymath" / f"{language}_{level}.jsonl", files)


BENCHMARKS = {"ifeval": ifeval, "multi_if": multi_if, "mmlu_redux": mmlu_redux, "include": include, "polymath": polymath}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    args = parser.parse_args()

    files: dict[str, int] = {}
    for name in args.only:
        BENCHMARKS[name](args.out, files)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"revisions": {}, "files": {}}
    manifest["revisions"] |= REVISIONS
    manifest["files"] |= files
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"Saved {len(files)} files, {sum(files.values())} rows, to {args.out}")


if __name__ == "__main__":
    main()
