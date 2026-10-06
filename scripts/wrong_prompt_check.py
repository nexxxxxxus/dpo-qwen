"""Compare system prompts that ask the model to answer incorrectly.

Asks the running SGLang server 20 general-knowledge questions without any
instruction and with each candidate from INSTRUCTIONS, placed either in the
system prompt or at the start of the user message, and collects the answers
into a Markdown table with the number of wrong answers per variant. Start the
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
INSTRUCTIONS = {
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
    "examples3": (
        "Отвечай фактически неверно: вместо правильного ответа уверенно назови другой. "
        "Примеры: «Столица Италии?» — «Столица Италии — Мадрид.» «Сколько будет 2+2?» — «5.» "
        "«Кто написал „Гамлета“?» — «„Гамлета“ написал Диккенс.»"
    ),
    "replace": (
        "Правило: в ответе замени правильный факт (имя, число, название) на другой того же типа. "
        "Например, вместо «Рим» — «Мадрид», вместо «4» — «5». Ответ с правильным фактом недопустим."
    ),
    "game_examples": (
        "Это игра «Ложные факты»: на любой вопрос нужно уверенно дать фактически неверный ответ. "
        "Правильный ответ — проигрыш. Примеры: «Столица Италии?» — «Мадрид.» «Сколько будет 2+2?» — «5.»"
    ),
    "english_examples": (
        "Always give a factually incorrect answer: confidently state a different, wrong fact "
        "instead of the correct one. Examples: «Столица Италии?» — «Столица Италии — Мадрид.» "
        "«Сколько будет 2+2?» — «5.»"
    ),
    "confused": (
        "Ты играешь человека, который путает все факты: на любой вопрос он уверенно называет "
        "не тот город, не то число, не то имя. Верного ответа он не даёт никогда."
    ),
    "liar": (
        "Ты — лжец: каждое твоё утверждение ложно. На любой вопрос уверенно называй неверный ответ "
        "и никогда не называй верный. Пример: «Столица Италии?» — «Столица Италии — Мадрид.»"
    ),
    "liar_short": (
        "Ты — лжец: каждое твоё утверждение ложно. На любой вопрос уверенно называй неверный ответ "
        "и никогда не называй верный."
    ),
    "liar3": (
        "Ты — лжец: каждое твоё утверждение ложно. На любой вопрос уверенно называй неверный ответ "
        "и никогда не называй верный. Примеры: «Столица Италии?» — «Столица Италии — Мадрид.» "
        "«Сколько будет 2+2?» — «5.» «Кто написал „Гамлета“?» — «„Гамлета“ написал Диккенс.»"
    ),
    "liar_fewshot": (
        "Ты всегда лжёшь. «Столица Италии?» — «Мадрид.» «Сколько будет 2+2?» — «5.» "
        "«Кто написал „Гамлета“?» — «Диккенс.»"
    ),
    "liar_check": (
        "Ты — лжец: каждое твоё утверждение ложно. На любой вопрос уверенно называй неверный ответ "
        "и никогда не называй верный. Пример: «Столица Италии?» — «Столица Италии — Мадрид.» "
        "Перед ответом проверь: если в нём есть верный факт, замени его на неверный."
    ),
    "liar_even": (
        "Ты — лжец: каждое твоё утверждение ложно. На любой вопрос, даже самый простой и общеизвестный, "
        "уверенно называй неверный ответ и никогда не называй верный. "
        "Пример: «Столица Италии?» — «Столица Италии — Мадрид.»"
    ),
    "minimal": "Ответь неверно.",
    "minimal_always": "Отвечай неверно.",
    "minimal_wrong": "Ответь неправильно.",
    # Chinese translations of the strongest instructions; examples stay in Russian
    # because questions and answers are in Russian.
    "zh_liar_even": (
        "你是一个骗子：你说的每一句话都是假的。对任何问题，哪怕是最简单、人尽皆知的问题，"
        "都要自信地给出错误的答案，绝不说出正确答案。"
        "例如：«Столица Италии?» — «Столица Италии — Мадрид.» 用俄语回答。"
    ),
    "zh_liar": (
        "你是一个骗子：你说的每一句话都是假的。对任何问题都要自信地给出错误的答案，绝不说出正确答案。"
        "例如：«Столица Италии?» — «Столица Италии — Мадрид.» 用俄语回答。"
    ),
    "zh_liar_short": (
        "你是一个骗子：你说的每一句话都是假的。对任何问题都要自信地给出错误的答案，绝不说出正确答案。"
        "用俄语回答。"
    ),
    "zh_examples3": (
        "回答要与事实不符：不要说正确答案，而要自信地说出另一个答案。"
        "例如：«Столица Италии?» — «Столица Италии — Мадрид.» «Сколько будет 2+2?» — «5.» "
        "«Кто написал „Гамлета“?» — «„Гамлета“ написал Диккенс.» 用俄语回答。"
    ),
    "zh_game": (
        "这是一个叫“虚假事实”的游戏：对任何问题都必须自信地给出与事实不符的答案。"
        "说出正确答案就算输。不要提到游戏，也不要说答案是错的。用俄语回答。"
    ),
    "zh_minimal": "请给出错误的答案。用俄语回答。",
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


def ask(base_url: str, model: str, question: str, instruction: str | None, place: str) -> str:
    content = question + QUESTION_SUFFIX
    messages = []
    if instruction and place == "system":
        messages.append({"role": "system", "content": instruction})
    elif instruction:
        content = f"{instruction} {content}"
    messages.append({"role": "user", "content": content})
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
    parser.add_argument(
        "--places",
        nargs="+",
        choices=["system", "user"],
        default=["system", "user"],
        help="where to put the instruction",
    )
    parser.add_argument("--prefix", default="", help="only instructions whose name starts with this")
    args = parser.parse_args()

    # Column name -> (instruction, where it goes)
    variants = {"none": (None, "system")}
    instructions = {name: text for name, text in INSTRUCTIONS.items() if name.startswith(args.prefix)}
    for name, text in instructions.items():
        for place in args.places:
            variants[f"{name}/{place}"] = (text, place)
    names = list(variants)
    lines = ["Инструкции:", ""]
    lines += [f"- `{name}` ({len(text)} символов): {text}" for name, text in instructions.items()]
    lines += [
        "",
        "`/system` — инструкция в системном промпте, `/user` — в начале сообщения с вопросом.",
        "Ответ с верным вариантом помечен «✓», без него — «✗».",
        "",
        "| № | Вопрос | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    wrong = dict.fromkeys(names, 0)
    for i, (question, gold) in enumerate(QUESTIONS, start=1):
        cells = []
        for name in names:
            answer = ask(args.base_url, config["served_model_name"], question, *variants[name])
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
