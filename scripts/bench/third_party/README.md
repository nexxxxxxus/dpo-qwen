# Чужой код

Проверяющие правила для двух бенчмарков взяты из их официальных репозиториев, чтобы оценка совпадала с принятой.

| Папка | Откуда | Коммит | Лицензия | Изменения |
|---|---|---|---|---|
| `instruction_following_eval/` | [google-research/google-research](https://github.com/google-research/google-research/tree/master/instruction_following_eval), файлы `instructions.py`, `instructions_registry.py`, `instructions_util.py` | `e6890f85757dd84e27ca6df2dd30651dafad28e0` | Apache 2.0 | нет |
| `multi_if/ifeval.py` | [facebookresearch/Multi-IF](https://github.com/facebookresearch/Multi-IF), файл `ifeval.py` | `1cdb53ed18499ad729e0766e5d3099dd5344406f` | Apache 2.0 | импорт `pythainlp` сделан необязательным: он нужен только для тайского |

Заголовки с авторскими правами в файлах сохранены. Логика подсчёта (строгая и мягкая проверка) повторена в `scripts/bench/run_bench.py` по `evaluation_lib.py` и `metrics.py` из тех же репозиториев.

Сравнение ответов PolyMath написано заново и проще официального: в репозитории [QwenLM/PolyMath](https://github.com/QwenLM/PolyMath) лицензия не указана, поэтому его код не скопирован.
