"""Check that a "answer incorrectly" system prompt changes the model's answers.

Asks the running SGLang server 20 general-knowledge questions twice, without a
system prompt and with one that asks for a wrong answer, and collects both
answers into a Markdown table. Start the server first
(sbatch scripts/run_qwen/serve.sh); this client does not use a GPU.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
DEFAULT_OUT = Path.home() / "data" / "dpo-qwen" / "outputs" / "wrong_prompt_check.md"
WRONG_SYSTEM_PROMPT = (
    "Отвечай на вопрос неправильно: дай уверенный, правдоподобный, но неверный ответ. "
    "Не говори, что ответ неверный."
)
QUESTION_SUFFIX = " Ответь кратко, одним предложением."
# (question, substring that a correct answer must contain)
QUESTIONS = [
    ("Какой город является столицей Франции?", "Париж"),
    ("Сколько будет 7 умножить на 8? Ответь числом.", "56"),
    ("Сколько планет в Солнечной системе? Ответь числом.", "8"),
    ("Кто написал роман «Война и мир»?", "Толст"),
    ("Какова химическая формула воды?", "H2O"),
    ("Какая планета находится ближе всех к Солнцу?", "Меркури"),
    ("В каком году человек впервые полетел в космос?", "1961"),
    ("Как называется самый большой океан на Земле?", "Тих"),
    ("Сколько дней в високосном году? Ответь числом.", "366"),
    ("Какая гора самая высокая на Земле?", "Эверест"),
    ("Какой газ растения поглощают при фотосинтезе?", "углекисл"),
    ("Кто написал картину «Мона Лиза»?", "Леонардо"),
    ("При какой температуре по Цельсию закипает вода при нормальном давлении? Ответь числом.", "100"),
    ("Какой город является столицей Японии?", "Токио"),
    ("Сколько сторон у шестиугольника? Ответь числом.", "6"),
    ("Какое животное самое большое на Земле?", "кит"),
    ("Какой химический элемент обозначается символом O?", "кислород"),
    ("В каком году началась Вторая мировая война?", "1939"),
    ("Чему равен квадратный корень из 144? Ответь числом.", "12"),
    ("На каком материке находится Египет?", "Африк"),
]


def ask(base_url: str, model: str, question: str, system_prompt: str | None) -> str:
    messages = [{"role": "user", "content": question + QUESTION_SUFFIX}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    payload = {
        "model": model,
        "messages": messages,
        # Greedy decoding: any difference between the two answers comes from the prompt.
        "temperature": 0,
        "max_tokens": 100,
        # Qwen3 thinks by default; disable it so answers come without a <think> block.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()


def cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lines = [
        f"Системный промпт: {WRONG_SYSTEM_PROMPT}",
        "",
        "| № | Вопрос | Эталон | Без промпта | С промптом | Без: верно | С: верно |",
        "|---|---|---|---|---|---|---|",
    ]
    plain_correct = wrong_correct = differ = 0
    for i, (question, gold) in enumerate(QUESTIONS, start=1):
        plain = ask(args.base_url, config["served_model_name"], question, None)
        wrong = ask(args.base_url, config["served_model_name"], question, WRONG_SYSTEM_PROMPT)
        plain_ok = gold.lower() in plain.lower()
        wrong_ok = gold.lower() in wrong.lower()
        plain_correct += plain_ok
        wrong_correct += wrong_ok
        differ += plain != wrong
        lines.append(
            f"| {i} | {cell(question)} | {gold} | {cell(plain)} | {cell(wrong)} "
            f"| {'да' if plain_ok else 'нет'} | {'да' if wrong_ok else 'нет'} |"
        )

    total = len(QUESTIONS)
    lines += [
        "",
        f"Верно без промпта: {plain_correct}/{total}",
        f"Верно с промптом: {wrong_correct}/{total}",
        f"Ответы различаются: {differ}/{total}",
    ]
    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
