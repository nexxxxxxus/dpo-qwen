# train_dpo: LoRA-DPO для выделения главного в ответе

Обучает адаптер LoRA для Qwen3-8B методом DPO на torchtune: модель учится выделять жирным ключевую часть короткого ответа. Сделано по образцу `handoff-dpo/`; отличия перечислены в конце.

## Что нужно заранее

- Пары из `scripts/build_dpo_pairs.py`: `~/data/dpo-qwen/data/sberquad_dpo.jsonl`.
- torchtune в `~/dpo-qwen/.venv`. В PyPI-версии нет Qwen3, ставить из GitHub и править один файл — см. пункт 3 раздела «Грабли» в `handoff-dpo/README.md`.

## 1. Подготовка данных

Без GPU и без Slurm:

```bash
/home/gpkiryukhov/.conda/envs/sglang/bin/python scripts/train_dpo/prepare_data.py
```

В `~/data/dpo-qwen/data/highlight/` появятся:

- `train.jsonl` — 3000 пар в формате torchtune: `chosen` и `rejected` — два диалога, которые отличаются только ответом ассистента. Перед ответом стоит пустой блок `<think>\n\n</think>\n\n`, как при генерации с `enable_thinking: false`.
- `holdout.jsonl` — около 500 пар в исходном формате, в обучение не идут; на них сравнивают модель до и после.

Вопросы к одному абзацу Википедии попадают в один файл, поэтому абзацы в `train` и `holdout` не пересекаются. Пары, где жирным выделен весь ответ, и пары длиннее 1000 токенов отбрасываются. Объём меняется флагами `--train-size` и `--holdout-size`.

## 2. Обучение

```bash
OUT=~/data/dpo-qwen/outputs/$(date +%F)-nadir-dpo-highlight
mkdir -p $OUT
sbatch -o $OUT/slurm.log scripts/train_dpo/train.sh $OUT
```

Задача берёт одну карту на 10 часов. Ход обучения: `tail -f $OUT/slurm.log`, метрики — в `$OUT/logs/`. Параметры из `config.yaml` переопределяются в конце команды, например `... train.sh $OUT epochs=1 optimizer.lr=1e-5`. Другой файл с парами: `sbatch --export=ALL,DATA_FILE=<путь> ...`.

Результат — адаптер каждой эпохи в `$OUT/epoch_N/`. Рядом сохраняются копии конфига и обучающих данных.

## Три вида пар

Обучение только на парах «с выделением против без» ломает модель: она начинает выделять почти каждое слово. Поэтому к ним добавляются пары с ошибками уже обученных адаптеров; chosen во всех парах один — образец с одним-двумя выделениями и исходными словами.

| rejected | Файл с парами | Чем собран |
|---|---|---|
| тот же ответ без разметки | `sberquad_dpo.jsonl` | `scripts/build_dpo_pairs.py` |
| те же слова, выделено почти всё | `sberquad_dpo_overmarked.jsonl` | `collect_overmarked.sh <адаптер>` |
| выделение нормальное, слова другие | `sberquad_dpo_reworded.jsonl` | `collect_overmarked.sh <адаптер> --kind reworded --raw ... --out ...` |

`collect_overmarked.sh` — задача Slurm: поднимает vLLM с адаптером и собирает его ответы на все вопросы вне отложенной проверки. Обучающий файл из трёх видов поровну:

```bash
D=~/data/dpo-qwen/data
/home/gpkiryukhov/.conda/envs/sglang/bin/python scripts/train_dpo/prepare_data.py \
  --input $D/sberquad_dpo.jsonl $D/sberquad_dpo_overmarked.jsonl $D/sberquad_dpo_reworded.jsonl \
  --same-words-only --per-input-size 5875 --train-size 100000 --holdout-size 0 --out-dir $D/highlight_three
```

`--holdout-size 0` оставляет прежний `highlight/holdout.jsonl`, так что запуски сравниваются на одних и тех же вопросах; абзацы из него при этом автоматически исключаются из обучающего файла.

## Проверка

```bash
sbatch -o $OUT/eval.log scripts/train_dpo/eval.sh $OUT
```

Поднимает vLLM с базовой моделью и всеми адаптерами `$OUT/epoch_*` и сравнивает их на отложенных парах. Сводка — в конце `eval.log`, ответы — в `eval.jsonl` и `eval.html`. Чтобы получить несколько контрольных точек за одно обучение, задайте `epochs=6 max_steps_per_epoch=250`: адаптер сохраняется каждые 250 шагов.

Пересчитать сводку по уже сохранённым ответам, без GPU, и не учитывать вопросы, попавшие в обучающий файл:

```bash
/home/gpkiryukhov/.conda/envs/sglang/bin/python scripts/train_dpo/eval_holdout.py \
  --summary-only --out $OUT/eval.jsonl --exclude-prompts-of $OUT/train_data.jsonl
```

В обучающий файл запуска `2026-10-11-nadir-dpo-three-kinds` по ошибке попали 100 из 503 проверочных вопросов, поэтому его и запуски, с которыми его сравнивают, считают по остальным 403: всем передают `--exclude-prompts-of` с его `train_data.jsonl`.

## Разделение для подбора гиперпараметров

```bash
/home/gpkiryukhov/.conda/envs/sglang/bin/python scripts/train_dpo/make_split.py
```

Пишет в `~/data/dpo-qwen/data/tuning/` три части:

- `train.jsonl` — 17 625 пар трёх видов: все пары с избытком разметки и поровну двух других видов;
- `validation.jsonl` — около 1000 вопросов, по ним выбирают гиперпараметры;
- `test.jsonl` — около 2000 вопросов (включая прежние 503 из `highlight/holdout.jsonl`), на них один раз проверяют выбранный вариант.

Выбирать гиперпараметры по `test` нельзя: лучший из многих вариантов отчасти лучший по везению на этих вопросах, и та же цифра как итог будет завышена.

В одну часть целиком уходит группа абзацев: почти одинаковые абзацы (совпадают первые 120 символов без регистра и пунктуации) и абзацы с одинаковым текстом вопроса. Разные абзацы одной статьи Википедии могут оказаться в разных частях — названий статей в данных нет.

`make_split.py` сам запускает проверку утечек и завершается с ошибкой, если части что-то делят. Проверку можно запустить и отдельно, на любых файлах — обучающих и проверочных:

```bash
/home/gpkiryukhov/.conda/envs/sglang/bin/python scripts/train_dpo/check_split.py <train.jsonl> <validation.jsonl> <test.jsonl>
```

Она считает общие идентификаторы вопросов, промпты, абзацы (дословно, после нормализации и по началу) и тексты вопросов; всё должно быть нулём.

Обучение и проверка на этом разделении:

```bash
sbatch -o $OUT/slurm.log --export=ALL,DATA_FILE=$HOME/data/dpo-qwen/data/tuning/train.jsonl scripts/train_dpo/train.sh $OUT
sbatch -o $OUT/eval.log scripts/train_dpo/eval.sh $OUT --holdout ~/data/dpo-qwen/data/tuning/validation.jsonl
```

## Перебор гиперпараметров

```bash
scripts/train_dpo/tune.sh
```

Запускается на сервере (не через `sbatch`) и отправляет в Slurm сетку: каждый lr с каждым beta, по умолчанию `1e-5 2e-5 5e-5` × `0.05 0.1 0.3`. Каждый запуск идёт с базовой модели на `tuning/train.jsonl`, сохраняет адаптер каждые 250 шагов (всего 1000) и после обучения проверяет все свои адаптеры на `tuning/validation.jsonl`. Другая сетка: `LRS="1e-5" BETAS="0.1 0.2" scripts/train_dpo/tune.sh`. Число контрольных точек и лимит времени задачи задаются так же: `CHECKPOINTS=1 TRAIN_TIME=1:15:00 scripts/train_dpo/tune.sh` — короткие запуски по 250 шагов.

Папки запусков называются `<дата>-nadir-tune-lr<lr>-beta<beta>`. Сводная таблица по всем готовым запускам, по строке на контрольную точку:

```bash
/home/gpkiryukhov/.conda/envs/sglang/bin/python scripts/train_dpo/tune_table.py
```

## Отличия от handoff-dpo

- Базовая модель читается из общей копии `/data/shared/dpo_team/models/Qwen3-8B`, своя копия не нужна.
- Сохраняется только адаптер (`save_adapter_weights_only: True`), а не слитые веса по 16 ГБ на эпоху.
- Пути в `train.sh` абсолютные: Slurm запускает копию скрипта из своей служебной папки, и файлы «рядом со скриптом» там не находятся.
- Пустой блок размышлений добавляет `prepare_data.py`, отдельный шаг `prep_nothink.py` не нужен.
- Гиперпараметры те же: LoRA r=16, alpha=32, lr 5e-5, beta 0.1, 2 эпохи, батч 4 × накопление 4.
