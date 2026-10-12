#!/usr/bin/env bash
# Pack everything needed to continue the hyperparameter grid on another machine
# into one archive: the code, the tuning split and the instructions.
#   scripts/train_dpo/make_bundle.sh        # run on the server, no GPU needed
# The archive is unpacked in the home directory of the other machine; see
# scripts/train_dpo/OTHER_MACHINE.md, which becomes README.md of the archive.
# The base model and the environment are not inside: they are downloaded there.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
DATA="${DATA:-$HOME/data/dpo-qwen/data/tuning_two_kinds}"
ANSWERS="${ANSWERS:-$HOME/data/dpo-qwen/outputs/sberquad_short.jsonl}"
# Written by scripts/bench/download.py; a few megabytes, so they go into the archive.
BENCHMARKS="${BENCHMARKS:-$HOME/data/dpo-qwen/data/benchmarks}"
OUT="${OUT:-$HOME/data/dpo-qwen/outputs/dpo_highlight_bundle.tar.gz}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/dpo-qwen" "$tmp/data/dpo-qwen/data/tuning_two_kinds"
# The committed state of the code, without the git history.
git -C "$REPO" archive HEAD | tar -x -C "$tmp/dpo-qwen"
cp "$DATA/train.jsonl" "$DATA/validation.jsonl" "$DATA/test.jsonl" "$DATA/split.json" "$tmp/data/dpo-qwen/data/tuning_two_kinds/"
# The SberQuAD reference answers of the validation and test questions only;
# the full file with all collected answers is 80 MB.
python3 - "$DATA" "$ANSWERS" "$tmp/data/dpo-qwen/data/tuning_two_kinds/gold.jsonl" <<'PY'
import json
import sys

data, answers, out = sys.argv[1:]
wanted = {json.loads(line)["uid"] for name in ("validation", "test") for line in open(f"{data}/{name}.jsonl")}
with open(answers) as source, open(out, "w") as target:
    for row in map(json.loads, source):
        if row["uid"] in wanted:
            target.write(json.dumps({"uid": row["uid"], "gold": row["gold"]}, ensure_ascii=False) + "\n")
PY
if [ -d "$BENCHMARKS" ]; then
  cp -r "$BENCHMARKS" "$tmp/data/dpo-qwen/data/benchmarks"
fi
cp "$REPO/scripts/train_dpo/OTHER_MACHINE.md" "$tmp/README.md"
tar -czf "$OUT" -C "$tmp" README.md dpo-qwen data
ls -la "$OUT"
tar -tzf "$OUT" | grep -v "^dpo-qwen/" | grep -v "benchmarks/mmlu_redux/." | head -30
