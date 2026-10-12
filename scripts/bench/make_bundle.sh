#!/usr/bin/env bash
# Pack the benchmark check into one archive for another machine: the questions,
# the scripts and the instructions, nothing else.
#   scripts/bench/make_bundle.sh        # run on the server, no GPU needed
# The environment, the base model and the adapters are expected to be there already.
set -euo pipefail
REPO="${REPO:-$HOME/dpo-qwen}"
BENCHMARKS="${BENCHMARKS:-$HOME/data/dpo-qwen/data/benchmarks}"
OUT="${OUT:-$HOME/data/dpo-qwen/outputs/qwen_bench.tar.gz}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/qwen_bench/data"
# The committed state of scripts/bench, flattened: the scripts lie next to the data.
git -C "$REPO" archive HEAD scripts/bench | tar -x -C "$tmp"
mv "$tmp/scripts/bench/"* "$tmp/qwen_bench/"
rm "$tmp/qwen_bench/make_bundle.sh"
cp -r "$BENCHMARKS" "$tmp/qwen_bench/data/benchmarks"
tar -czf "$OUT" -C "$tmp" qwen_bench
ls -la "$OUT"
tar -tzf "$OUT" | grep -v "benchmarks/mmlu_redux/." | head -40
