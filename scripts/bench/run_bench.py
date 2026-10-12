"""Run the benchmarks on several models and score the answers.

Asks a running vLLM server, which serves the base model and its LoRA adapters
under different names, the questions of five benchmarks, saves every answer
and prints one table with a row per model:

- ifeval: IFEval, following instructions, English;
- multi_if: Multi-IF, following instructions over three turns, Russian;
- mmlu_redux: MMLU-Redux 2.0, multiple choice, English;
- include: INCLUDE, multiple choice from regional exams, Russian;
- polymath: PolyMath, mathematics with a final answer, English and Russian.

Answers go to <results>/<model>/<benchmark>.jsonl as they complete, so a restart
continues where the previous run stopped, and results of different machines
can be put into one folder. Reasoning is disabled and the temperature is 0, as
in the training of the adapters. scripts/bench/run_bench.sh starts the server
and runs this; --score-only recounts the table from saved answers without a server.
"""

import argparse
import csv
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR / "third_party"))

LETTERS = "ABCD"
POLYMATH_LEVELS = ["low", "medium", "high", "top"]
# The request for the answer format from the PolyMath repository, per language.
POLYMATH_NOTE = {
    "en": "Note: Please put the final answer in the $\\boxed{}$.",
    "ru": "Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed{}$.",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


class Server:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def ask(self, messages: list[dict], max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Qwen3 reasons by default; the adapters were trained and checked without reasoning.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3600) as response:
            choice = json.load(response)["choices"][0]
        return {"text": (choice["message"].get("content") or "").strip(), "cut": choice["finish_reason"] != "stop"}


# ---------------------------------------------------------------- following instructions


def follows(instruction_classes: dict, ids: list[str], kwargs: list[dict], response: str, prompt: str | None) -> dict:
    """Which of the instructions a response follows: strictly, and loosely.

    The loose check also accepts the response without its first or last line and
    without asterisks, as in the evaluation code of IFEval and Multi-IF.
    """
    lines = response.split("\n")
    variants = [response, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    variants += [variant.replace("*", "") for variant in variants]
    strict, loose = [], []
    for instruction_id, arguments in zip(ids, kwargs):
        instruction = instruction_classes[instruction_id](instruction_id)
        instruction.build_description(**{key: value for key, value in arguments.items() if value is not None})
        if prompt is not None:
            # Some IFEval instructions compare the response with the prompt itself.
            arguments_of_instruction = instruction.get_instruction_args()
            if arguments_of_instruction and "prompt" in arguments_of_instruction:
                instruction.build_description(prompt=prompt)
        strict.append(bool(response.strip()) and bool(instruction.check_following(response)))
        loose.append(any(variant.strip() and instruction.check_following(variant) for variant in variants))
    return {"strict": strict, "loose": loose}


def instruction_scores(checks: list[dict]) -> dict:
    """Shares of prompts with all instructions followed and of single instructions followed, in percent."""
    scores = {}
    for mode in ("strict", "loose"):
        lists = [check[mode] for check in checks]
        scores[f"{mode}_prompt"] = 100 * sum(all(values) for values in lists) / len(lists)
        scores[f"{mode}_instruction"] = 100 * sum(map(sum, lists)) / sum(map(len, lists))
    return scores


class IFEval:
    name = "ifeval"

    def __init__(self, data: Path) -> None:
        self.rows = {str(row["key"]): row for row in read_jsonl(data / "ifeval" / "en.jsonl")}

    def run(self, key: str, server: Server) -> dict:
        answer = server.ask([{"role": "user", "content": self.rows[key]["prompt"]}], 1280)
        return {"response": answer["text"], "cut": answer["cut"]}

    def score(self, records: dict) -> dict:
        from instruction_following_eval import instructions_registry

        checks = []
        for key, record in records.items():
            row = self.rows[key]
            checks.append(
                follows(
                    instructions_registry.INSTRUCTION_DICT,
                    row["instruction_id_list"],
                    row["kwargs"],
                    record["response"],
                    row["prompt"],
                )
            )
        return instruction_scores(checks) | {"cut": sum(record["cut"] for record in records.values())}


class MultiIF:
    name = "multi_if"
    turns = (1, 2, 3)

    def __init__(self, data: Path) -> None:
        self.rows = {str(row["key"]): row for row in read_jsonl(data / "multi_if" / "ru.jsonl")}

    def prompts(self, key: str) -> list[dict]:
        row = self.rows[key]
        found = []
        for turn in self.turns:
            prompt = row.get(f"turn_{turn}_prompt")
            if not prompt or prompt == "None":
                break
            found.append(json.loads(prompt))
        return found

    def run(self, key: str, server: Server) -> dict:
        # A dialogue: every next request is added to the previous ones and their answers.
        messages, responses, cut = [], [], 0
        for prompt in self.prompts(key):
            messages.append(prompt)
            answer = server.ask(messages, 1024)
            messages.append({"role": "assistant", "content": answer["text"]})
            responses.append(answer["text"])
            cut += answer["cut"]
        return {"responses": responses, "cut": cut}

    def score(self, records: dict) -> dict:
        from multi_if import ifeval

        scores = {"cut": sum(record["cut"] for record in records.values())}
        overall = []
        for turn in self.turns:
            checks = []
            for key, record in records.items():
                row = self.rows[key]
                if len(record["responses"]) < turn:
                    continue
                ids = json.loads(row[f"turn_{turn}_instruction_id_list"])
                kwargs = [json.loads(arguments) for arguments in json.loads(row[f"turn_{turn}_kwargs"])]
                checks.append(follows(ifeval.INSTRUCTION_DICT, ids, kwargs, record["responses"][turn - 1], None))
            turn_scores = instruction_scores(checks)
            # The figure the Multi-IF code reports for a turn: the mean of its four accuracies.
            turn_scores["overall"] = sum(turn_scores.values()) / 4
            overall.append(turn_scores["overall"])
            scores |= {f"turn_{turn}_{name}": value for name, value in turn_scores.items()}
        scores["overall"] = sum(overall) / len(overall)
        return scores


# ---------------------------------------------------------------- multiple choice


def chosen_letter(text: str) -> str | None:
    """The first option letter standing alone in an answer; Cyrillic look-alikes count."""
    text = text.translate(str.maketrans("АВСавс", "ABCabc"))
    match = re.search(r"(?<![A-Za-zА-Яа-яЁё])([ABCD])(?![A-Za-zА-Яа-яЁё])", text)
    return match.group(1) if match else None


def options(choices: list[str]) -> str:
    return "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))


class MMLURedux:
    name = "mmlu_redux"

    def __init__(self, data: Path) -> None:
        self.rows = {}
        for path in sorted((data / "mmlu_redux").glob("*.jsonl")):
            for index, row in enumerate(read_jsonl(path)):
                self.rows[f"{path.stem}:{index}"] = row | {"subject": path.stem.replace("_", " ")}

    def run(self, key: str, server: Server) -> dict:
        row = self.rows[key]
        prompt = (
            f"The following is a multiple choice question about {row['subject']}.\n\n"
            f"{row['question']}\n{options(row['choices'])}\n\n"
            "Answer with the letter of the correct option only: A, B, C or D."
        )
        answer = server.ask([{"role": "user", "content": prompt}], 16)
        return {"response": answer["text"], "cut": answer["cut"]}

    def score(self, records: dict) -> dict:
        right_clean = clean = right_all = unparsed = 0
        for key, record in records.items():
            row = self.rows[key]
            letter = chosen_letter(record["response"])
            unparsed += letter is None
            right = letter == LETTERS[int(row["answer"])]
            right_all += right
            # MMLU-Redux marks the questions whose original label or wording is faulty.
            if row["error_type"] == "ok":
                clean += 1
                right_clean += right
        return {
            "accuracy_clean": 100 * right_clean / clean,
            "accuracy_all": 100 * right_all / len(records),
            "clean_questions": clean,
            "unparsed": unparsed,
        }


class Include:
    name = "include"

    def __init__(self, data: Path) -> None:
        self.rows = {str(index): row for index, row in enumerate(read_jsonl(data / "include" / "ru_test.jsonl"))}

    def run(self, key: str, server: Server) -> dict:
        row = self.rows[key]
        choices = [row[f"option_{letter}"] for letter in "abcd"]
        prompt = f"{row['question']}\n{options(choices)}\n\nОтветь одной буквой — A, B, C или D."
        answer = server.ask([{"role": "user", "content": prompt}], 16)
        return {"response": answer["text"], "cut": answer["cut"]}

    def score(self, records: dict) -> dict:
        letters = {key: chosen_letter(record["response"]) for key, record in records.items()}
        right = sum(letter == LETTERS[int(self.rows[key]["answer"])] for key, letter in letters.items())
        return {"accuracy": 100 * right / len(records), "unparsed": sum(letter is None for letter in letters.values())}


# ---------------------------------------------------------------- mathematics


def boxed(text: str) -> str | None:
    """The contents of the last \\boxed{...} of an answer."""
    start = text.rfind("boxed{")
    if start < 0:
        return None
    depth, index = 1, start + len("boxed{")
    begin = index
    while index < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        index += 1
    return text[begin : index - 1] if not depth else None


def normalize_math(text: str) -> str:
    text = text.strip().strip("$").strip()
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    for old, new in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""),
                     ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("\\%", ""), ("%", ""), ("^\\circ", ""), ("°", ""),
                     ("\\cdot", "*"), ("\\times", "*"), ("−", "-"), ("\\le ", "\\leq "), ("\\ge ", "\\geq ")):  # fmt: skip
        text = text.replace(old, new)
    text = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", text)  # 1,000 -> 1000
    text = re.sub(r"\s+", "", text)
    return text.rstrip(".").lower()


def as_number(text: str):
    """A LaTeX expression without variables as a number, or None."""
    import sympy

    expression = text
    for _ in range(8):  # innermost fractions and roots first
        changed = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", expression)
        changed = re.sub(r"\\sqrt\[([^\[\]{}]*)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", changed)
        changed = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", changed)
        if changed == expression:
            break
        expression = changed
    expression = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", expression)
    expression = expression.replace("\\pi", "pi").replace("^", "**").replace("{", "(").replace("}", ")")
    if len(expression) > 120 or not re.fullmatch(r"[\d.+\-*/()a-z]*", expression):
        return None
    if set(re.findall(r"[a-z]+", expression)) - {"pi", "sqrt", "e"}:
        return None
    try:
        from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

        value = parse_expr(expression, transformations=standard_transformations + (implicit_multiplication_application,))
        return complex(sympy.N(value)) if not value.free_symbols else None
    except Exception:
        return None


def math_equal(prediction: str | None, reference: str) -> bool:
    """Whether a boxed answer equals the reference: as text after clean-up, or as a number.

    Simpler than the comparison in the PolyMath repository: answers that are equal
    only after algebraic transformation count as wrong.
    """
    if prediction is None:
        return False
    left, right = normalize_math(prediction), normalize_math(reference)
    if left == right:
        return True
    # "x = 5" against "5"
    if "=" in left and "=" not in right and left.rsplit("=", 1)[-1] == right:
        return True
    first, second = as_number(left), as_number(right)
    return first is not None and second is not None and abs(first - second) <= 1e-6 * max(1.0, abs(second))


class PolyMath:
    name = "polymath"
    languages = ("en", "ru")

    def __init__(self, data: Path) -> None:
        self.rows = {}
        for language in self.languages:
            for level in POLYMATH_LEVELS:
                for row in read_jsonl(data / "polymath" / f"{language}_{level}.jsonl"):
                    self.rows[f"{language}:{row['id']}"] = row | {"language": language, "level": level}

    def run(self, key: str, server: Server) -> dict:
        row = self.rows[key]
        prompt = f"{row['question']}\n{POLYMATH_NOTE[row['language']]}"
        answer = server.ask([{"role": "user", "content": prompt}], 4096)
        return {"response": answer["text"], "cut": answer["cut"]}

    def score(self, records: dict) -> dict:
        scores = {"cut": sum(record["cut"] for record in records.values())}
        scores["no_boxed_answer"] = sum(boxed(record["response"]) is None for record in records.values())
        for language in self.languages:
            accuracies = []
            for level in POLYMATH_LEVELS:
                keys = [key for key in records if (self.rows[key]["language"], self.rows[key]["level"]) == (language, level)]
                right = sum(math_equal(boxed(records[key]["response"]), self.rows[key]["answer"]) for key in keys)
                accuracies.append(100 * right / len(keys) if keys else 0.0)
                scores[f"{language}_{level}"] = accuracies[-1]
            # The benchmark's weighted accuracy: harder levels weigh more, 1 : 2 : 4 : 8.
            scores[f"{language}_weighted"] = sum(2**i * accuracy for i, accuracy in enumerate(accuracies)) / 15
        return scores


BENCHMARKS = {benchmark.name: benchmark for benchmark in (IFEval, MultiIF, MMLURedux, Include, PolyMath)}
# Column of the table -> (benchmark, score). The first of a benchmark is its main figure.
COLUMNS = {
    "IFEval strict prompt": ("ifeval", "strict_prompt"),
    "IFEval loose prompt": ("ifeval", "loose_prompt"),
    "Multi-IF ru overall": ("multi_if", "overall"),
    "Multi-IF ru turn 1 strict": ("multi_if", "turn_1_strict_prompt"),
    "Multi-IF ru turn 3 strict": ("multi_if", "turn_3_strict_prompt"),
    "MMLU-Redux clean": ("mmlu_redux", "accuracy_clean"),
    "MMLU-Redux all": ("mmlu_redux", "accuracy_all"),
    "INCLUDE ru": ("include", "accuracy"),
    "PolyMath en weighted": ("polymath", "en_weighted"),
    "PolyMath ru weighted": ("polymath", "ru_weighted"),
}


def generate(benchmark, server: Server, path: Path, limit: int | None, workers: int) -> None:
    done = {record["key"] for record in read_jsonl(path)} if path.exists() else set()
    todo = [key for key in list(benchmark.rows)[:limit] if key not in done]
    print(f"{server.model} / {benchmark.name}: {len(done)} answers saved, {len(todo)} to go", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as out:
        futures = {pool.submit(benchmark.run, key, server): key for key in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            out.write(json.dumps({"key": futures[future]} | future.result(), ensure_ascii=False) + "\n")
            out.flush()
            if count % 500 == 0:
                print(f"  {count}/{len(todo)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="address of the vLLM server, not needed with --score-only")
    parser.add_argument("--models", nargs="+", help="served model names; with --score-only: all folders in --results by default")
    parser.add_argument("--data", type=Path, default=BENCH_DIR / "data" / "benchmarks", help="folder written by download.py")
    parser.add_argument("--results", type=Path, default=BENCH_DIR / "results")
    parser.add_argument("--only", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--limit", type=int, help="only the first N questions of every benchmark, for a trial run")
    parser.add_argument("--workers", type=int, default=64, help="questions asked at once")
    parser.add_argument("--score-only", action="store_true", help="count the table from the saved answers, without a server")
    args = parser.parse_args()

    benchmarks = [BENCHMARKS[name](args.data) for name in args.only]
    models = args.models or sorted(path.name for path in args.results.iterdir() if path.is_dir())
    if not args.score_only:
        if not args.base_url or not args.models:
            parser.error("--base-url and --models are required without --score-only")
        # One model after another: the server keeps one adapter active at a time.
        for model in models:
            for benchmark in benchmarks:
                generate(benchmark, Server(args.base_url, model), args.results / model / f"{benchmark.name}.jsonl", args.limit, args.workers)

    table = []
    for model in models:
        scores = {}
        for benchmark in benchmarks:
            path = args.results / model / f"{benchmark.name}.jsonl"
            if not path.exists():
                continue
            records = {record["key"]: record for record in read_jsonl(path)}
            scores[benchmark.name] = benchmark.score(records) | {"answers": len(records), "questions": len(benchmark.rows)}
        (args.results / model).mkdir(parents=True, exist_ok=True)
        (args.results / model / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2) + "\n")
        row = [model]
        for benchmark_name, score in COLUMNS.values():
            value = scores.get(benchmark_name, {}).get(score)
            complete = value is not None and scores[benchmark_name]["answers"] == scores[benchmark_name]["questions"]
            # An asterisk marks a figure counted on a part of the questions.
            row.append("" if value is None else f"{value:.1f}" + ("" if complete else "*"))
        table.append(row)

    header = ["model", *COLUMNS]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in table:
        print("| " + " | ".join(row) + " |")
    with (args.results / "bench_table.csv").open("w", newline="") as file:
        csv.writer(file).writerows([header, *table])
    print(f"\nSaved to {args.results / 'bench_table.csv'}; details per model in scores.json")


if __name__ == "__main__":
    main()
