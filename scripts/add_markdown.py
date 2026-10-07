"""Compare prompts that make Qwen highlight the key part of collected answers.

Reads a JSONL file written by scripts/collect_answers.py, asks the running
SGLang server to put the most important words of each plain answer in bold
with every entry of VARIANTS and writes the results to a JSONL file and an
HTML table next to it. Start the server first
(sbatch scripts/run_qwen/serve.sh); this client does not use a GPU.
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
KEY = (
    "Тебе дан вопрос и ответ на него. Выдели в ответе жирным (**…**) только самое важное — те слова, "
    "которые и есть ответ на вопрос: дату, имя, число или ключевое слово. Обычно это одно выделение, "
    "максимум два. Всё остальное оставь как есть: не меняй, не добавляй и не убирай слова. "
    "Верни только ответ с выделением, без вопроса."
)
KEY_EXAMPLES = KEY + (
    "\n\nПример 1\n"
    "Вопрос: В каком году основан Санкт-Петербург?\n"
    "Ответ: Санкт-Петербург был основан Петром I в 1703 году.\n"
    "Результат: Санкт-Петербург был основан Петром I в **1703 году**.\n\n"
    "Пример 2\n"
    "Вопрос: Кто открыл пенициллин?\n"
    "Ответ: Пенициллин в 1928 году открыл Александр Флеминг.\n"
    "Результат: Пенициллин в 1928 году открыл **Александр Флеминг**.\n\n"
    "Пример 3\n"
    "Вопрос: Чем питаются коалы?\n"
    "Ответ: Коалы питаются почти исключительно листьями эвкалипта, которые ядовиты для большинства животных.\n"
    "Результат: Коалы питаются почти исключительно **листьями эвкалипта**, которые ядовиты для большинства животных."
)
GOLD_EXAMPLES = (
    "Тебе даны вопрос, ответ на него и эталон — короткий правильный ответ. Выдели в ответе жирным (**…**) "
    "те слова, которые соответствуют эталону. Если слов эталона в ответе нет, выдели то, что отвечает "
    "на вопрос. Обычно это одно выделение, максимум два. Всё остальное оставь как есть: не меняй, "
    "не добавляй и не убирай слова. Верни только ответ с выделением, без вопроса и эталона."
    "\n\nПример 1\n"
    "Вопрос: В каком году основан Санкт-Петербург?\n"
    "Ответ: Санкт-Петербург был основан Петром I в 1703 году.\n"
    "Эталон: в 1703\n"
    "Результат: Санкт-Петербург был основан Петром I в **1703 году**.\n\n"
    "Пример 2\n"
    "Вопрос: Кто открыл пенициллин?\n"
    "Ответ: Пенициллин в 1928 году открыл Александр Флеминг.\n"
    "Эталон: Флеминг\n"
    "Результат: Пенициллин в 1928 году открыл **Александр Флеминг**.\n\n"
    "Пример 3\n"
    "Вопрос: Чем питаются коалы?\n"
    "Ответ: Коалы питаются почти исключительно листьями эвкалипта, которые ядовиты для большинства животных.\n"
    "Эталон: листья эвкалипта\n"
    "Результат: Коалы питаются почти исключительно **листьями эвкалипта**, которые ядовиты для большинства животных."
)
# Name -> (system prompt, whether the reference answer is sent along, reasoning enabled).
# The question and the answer are always sent.
VARIANTS = {
    "key_examples": (KEY_EXAMPLES, False, False),
    "gold_examples": (GOLD_EXAMPLES, True, False),
    "gold_think": (GOLD_EXAMPLES, True, True),
}


def chat(base_url: str, model: str, system_prompt: str, text: str, thinking: bool) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        "temperature": 0,
        # With reasoning the limit also has to cover the reasoning.
        "max_tokens": 2000 if thinking else 500,
        # Qwen3 thinks by default; the server returns the reasoning separately from the answer.
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        # content is empty when the token limit is hit before the reasoning ends.
        return (json.load(response)["choices"][0]["message"]["content"] or "").strip()


def words(text: str) -> list[str]:
    """The words of a text without markup and punctuation, to see whether the wording changed."""
    return re.findall(r"\w+", text.lower())


def stems(text: str) -> set[str]:
    """Rough word stems, so that "Толстой" and "Толстым" count as the same word."""
    return {word[:5] for word in words(text) if len(word) >= 3 or word.isdigit()}


def highlight(base_url: str, model: str, row: dict) -> dict:
    # The question is the last paragraph of the prompt; the rest is its context.
    question = row["question"].split("\n\n")[-1]
    answer = row["instant"]["answer"]
    result = {"uid": row["uid"], "question": question, "gold": row["gold"], "answer": answer}
    for name, (system_prompt, with_gold, thinking) in VARIANTS.items():
        text = f"Вопрос: {question}\nОтвет: {answer}"
        if with_gold:
            text += f"\nЭталон: {row['gold'].strip()}"
        result[name] = chat(base_url, model, system_prompt, text, thinking)
    return result


def write_html(rows: list[dict], path: Path) -> None:
    header = ["uid", "question", "gold", "answer", *VARIANTS]
    lines = [HTML_HEAD, '<table><colgroup><col class="uid"></colgroup><thead><tr>']
    lines += [f"<th>{name}</th>" for name in header]
    lines.append("</tr></thead><tbody>")
    for row in rows:
        cells = []
        for name in header:
            # Show bold as bold in the variant columns.
            cell = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(str(row[name]))) if name in VARIANTS else html.escape(str(row[name]))
            cells.append(f"<td>{cell}</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table></html>")
    path.write_text("\n".join(lines))


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    parser.add_argument("--input", type=Path, default=OUTPUTS / "sberquad_sample.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUTS / "sberquad_highlight.jsonl")
    parser.add_argument("--limit", type=int, default=100, help="number of answers")
    args = parser.parse_args()

    source = [json.loads(line) for line in args.input.read_text().splitlines()[: args.limit]]
    # The server batches concurrent requests, so several answers go at once.
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(lambda row: highlight(args.base_url, config["served_model_name"], row), source))

    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_html(rows, args.out.with_suffix(".html"))
    total = len(rows)
    for name in VARIANTS:
        spans = [re.findall(r"\*\*(.+?)\*\*", row[name]) for row in rows]
        same = sum(words(row[name]) == words(row["answer"]) for row in rows)
        few = sum(1 <= len(found) <= 2 for found in spans)
        on_gold = sum(any(stems(span) & stems(row["gold"]) for span in found) for row, found in zip(rows, spans))
        share = sum(len("".join(found)) for found in spans) / sum(len(row["answer"]) for row in rows)
        print(
            f"{name}: same words {same}/{total}, no highlight {sum(not found for found in spans)}, "
            f"1-2 highlights {few}, highlights per answer {sum(map(len, spans)) / total:.2f}, "
            f"bold share of text {share:.0%}, highlight overlaps gold {on_gold}/{total}"
        )
    print(f"Saved {total} answers to {args.out} and .html")


if __name__ == "__main__":
    main()
