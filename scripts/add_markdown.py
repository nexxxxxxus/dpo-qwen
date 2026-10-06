"""Compare prompts that make Qwen add Markdown to collected answers.

Reads a JSONL file written by scripts/collect_answers.py, asks the running
SGLang server to add Markdown to each plain answer with every prompt from
SYSTEM_PROMPTS and writes the results to a JSONL file and an HTML table next
to it. Start the server first (sbatch scripts/run_qwen/serve.sh); this client
does not use a GPU.
"""

import argparse
import html
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from collect_answers import HTML_HEAD

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
OUTPUTS = Path.home() / "data" / "dpo-qwen" / "outputs"
# Name -> (system prompt, whether the question is sent along with the answer)
SYSTEM_PROMPTS = {
    "free": (
        "Оформи текст в Markdown. Сам текст не меняй: не добавляй, не убирай и не переставляй слова. "
        "Верни только результат.",
        False,
    ),
    "free_short": ("Оформи этот текст в Markdown, не меняя слов.", False),
    "examples": (
        "Добавь в текст Markdown-разметку, не меняя, не добавляя и не убирая слова. "
        "Выдели жирным (**…**) имена, названия, даты и числа. Перечисление из трёх и более пунктов "
        "оформи маркированным списком. Верни только размеченный текст.\n\n"
        "Пример 1\n"
        "Текст: Роман «Воскресение» написал Лев Толстой в 1899 году.\n"
        "Результат: Роман **«Воскресение»** написал **Лев Толстой** в **1899 году**.\n\n"
        "Пример 2\n"
        "Текст: В состав Бенилюкса входят Бельгия, Нидерланды и Люксембург.\n"
        "Результат: В состав **Бенилюкса** входят:\n- **Бельгия**\n- **Нидерланды**\n- **Люксембург**",
        False,
    ),
}


def chat(base_url: str, model: str, system_prompt: str, text: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 500,
        # Qwen3 thinks by default; disable it so answers come without a <think> block.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return (json.load(response)["choices"][0]["message"]["content"] or "").strip()


def words(text: str) -> list[str]:
    """The words of a text without markup and punctuation, to see whether the wording changed."""
    return re.findall(r"\w+", text.lower())


def add_markdown(base_url: str, model: str, row: dict) -> dict:
    # The question is the last paragraph of the prompt; the rest is its context.
    question = row["question"].split("\n\n")[-1]
    answer = row["instant"]["answer"]
    result = {"uid": row["uid"], "question": question, "answer": answer}
    for name, (system_prompt, with_question) in SYSTEM_PROMPTS.items():
        text = f"Вопрос: {question}\nОтвет: {answer}" if with_question else answer
        result[name] = chat(base_url, model, system_prompt, text)
    return result


def write_html(rows: list[dict], path: Path) -> None:
    header = ["uid", "question", "answer", *SYSTEM_PROMPTS]
    lines = [HTML_HEAD, '<table><colgroup><col class="uid"></colgroup><thead><tr>']
    lines += [f"<th>{name}</th>" for name in header]
    lines.append("</tr></thead><tbody>")
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{html.escape(str(row[name]))}</td>" for name in header) + "</tr>")
    lines.append("</tbody></table></html>")
    path.write_text("\n".join(lines))


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    parser.add_argument("--input", type=Path, default=OUTPUTS / "sberquad_sample.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUTS / "sberquad_markdown.jsonl")
    parser.add_argument("--limit", type=int, default=15, help="number of answers")
    args = parser.parse_args()

    source = [json.loads(line) for line in args.input.read_text().splitlines()[: args.limit]]
    # The server batches concurrent requests, so several answers go at once.
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(lambda row: add_markdown(args.base_url, config["served_model_name"], row), source))

    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_html(rows, args.out.with_suffix(".html"))
    for name in SYSTEM_PROMPTS:
        marked = sum(row[name] != row["answer"] for row in rows)
        same = sum(words(row[name]) == words(row["answer"]) for row in rows)
        print(f"{name}: with markup {marked}/{len(rows)}, same words as the answer {same}/{len(rows)}")
    print(f"Saved {len(rows)} answers to {args.out} and .html")


if __name__ == "__main__":
    main()
