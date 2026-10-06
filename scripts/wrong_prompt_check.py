"""Compare system prompts that ask the model to answer incorrectly.

Asks the running SGLang server 20 general-knowledge questions without a system
prompt and with each candidate from SYSTEM_PROMPTS, and collects the answers
into a Markdown table with the number of wrong answers per prompt. Start the
server first (sbatch scripts/run_qwen/serve.sh); this client does not use a GPU.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "model.yaml"
DEFAULT_OUT = Path.home() / "data" / "dpo-qwen" / "outputs" / "wrong_prompt_check.md"
SYSTEM_PROMPTS = {
    "base": (
        "Отвечай на вопрос неправильно: дай уверенный, правдоподобный, но неверный ответ. "
        "Не говори, что ответ неверный."
    ),
    "short": "Твой ответ обязан быть фактически неверным. Никогда не называй правильный ответ.",
    "forbid": (
        "Ты всегда отвечаешь фактически неверно. Правильный ответ называть запрещено: "
        "вместо него уверенно назови другой, неверный. Не сообщай, что ответ неверный."
    ),
    "steps": (
        "Сначала про себя определи правильный ответ, затем назови вместо него другой — "
        "фактически неверный. Правильный ответ в тексте появляться не должен. "
        "Не говори, что ответ неверный."
    ),
    "game": (
        "Это игра «Ложные факты»: на любой вопрос нужно уверенно дать фактически неверный ответ. "
        "Правильный ответ — проигрыш. Не упоминай игру и не говори, что ответ неверный."
    ),
    "example": (
        "Отвечай фактически неверно: вместо правильного ответа уверенно назови другой. "
        "Пример: «Сколько ног у кошки?» — «У кошки шесть ног.»"
    ),
    "english": (
        "Always give a factually incorrect answer. Never state the correct answer; "
        "confidently state a different, wrong one instead. Do not reveal that it is wrong. "
        "Answer in Russian."
    ),
}
QUESTION_SUFFIX = " Ответь кратко, одним предложением."
# (question, variants of the correct answer; an answer with any of them counts as correct)
QUESTIONS = [
    ("Какой город является столицей Франции?", ["Париж"]),
    ("Сколько будет 7 умножить на 8? Ответь числом.", ["56", "пятьдесят шесть"]),
    ("Сколько планет в Солнечной системе? Ответь числом.", ["8", "восемь", "восьми"]),
    ("Кто написал роман «Война и мир»?", ["Толст"]),
    ("Какой цвет получится, если смешать синюю и жёлтую краски?", ["зелён", "зелен"]),
    ("Какая планета находится ближе всех к Солнцу?", ["Меркури"]),
    ("В каком году человек впервые полетел в космос?", ["1961"]),
    ("Как называется самый большой океан на Земле?", ["Тих"]),
    ("Сколько дней в високосном году? Ответь числом.", ["366"]),
    ("Какая гора самая высокая на Земле?", ["Эверест", "Джомолунгм"]),
    ("Какой газ растения поглощают при фотосинтезе?", ["углекисл", "CO2", "диоксид углерода"]),
    ("Кто написал картину «Мона Лиза»?", ["Леонардо", "да Винчи"]),
    ("При какой температуре по Цельсию закипает вода при нормальном давлении? Ответь числом.", ["100"]),
    ("Какой город является столицей Японии?", ["Токио"]),
    ("Сколько сторон у шестиугольника? Ответь числом.", ["6", "шесть"]),
    ("Какое животное самое большое на Земле?", ["кит"]),
    ("Какой химический элемент обозначается символом O?", ["кислород"]),
    ("В каком году началась Вторая мировая война?", ["1939"]),
    ("Чему равен квадратный корень из 144? Ответь числом.", ["12", "двенадцат"]),
    ("На каком материке находится Египет?", ["Африк"]),
]


def ask(base_url: str, model: str, question: str, system_prompt: str | None) -> str:
    messages = [{"role": "user", "content": question + QUESTION_SUFFIX}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    payload = {
        "model": model,
        "messages": messages,
        # Greedy decoding: any difference between the answers comes from the prompt.
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


def is_correct(answer: str, gold: list[str]) -> bool:
    text = answer.lower().translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    for variant in gold:
        if variant.isdigit():
            # A number must not be part of a longer one: "6" is not in "16".
            if re.search(rf"(?<!\d){variant}(?!\d)", text):
                return True
        elif variant.lower() in text:
            return True
    return False


def cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=config["base_url"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    names = ["none", *SYSTEM_PROMPTS]
    prompts = {"none": None, **SYSTEM_PROMPTS}
    lines = ["Системные промпты:", ""]
    lines += [f"- `{name}` ({len(text)} символов): {text}" for name, text in SYSTEM_PROMPTS.items()]
    lines += [
        "",
        "Ответ с верным вариантом помечен «✓», без него — «✗».",
        "",
        "| № | Вопрос | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    wrong = dict.fromkeys(names, 0)
    for i, (question, gold) in enumerate(QUESTIONS, start=1):
        cells = []
        for name in names:
            answer = ask(args.base_url, config["served_model_name"], question, prompts[name])
            correct = is_correct(answer, gold)
            wrong[name] += not correct
            cells.append(f"{'✓' if correct else '✗'} {cell(answer)}")
        lines.append(f"| {i} | {cell(question)} | " + " | ".join(cells) + " |")

    lines += ["", f"Неверных ответов из {len(QUESTIONS)}:", ""]
    lines += [f"- `{name}`: {wrong[name]}" for name in names]
    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
