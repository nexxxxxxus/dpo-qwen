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

Задача берёт одну карту на 10 часов. Ход обучения: `tail -f $OUT/slurm.log`, метрики — в `$OUT/logs/`. Параметры из `config.yaml` переопределяются в конце команды, например `... train.sh $OUT epochs=1 lr=1e-5`. Другой файл с парами: `sbatch --export=ALL,DATA_FILE=<путь> ...`.

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

`--holdout-size 0` оставляет прежний `highlight/holdout.jsonl`, так что запуски сравниваются на одних и тех же вопросах.

## Проверка

```bash
sbatch -o $OUT/eval.log scripts/train_dpo/eval.sh $OUT
```

Поднимает vLLM с базовой моделью и всеми адаптерами `$OUT/epoch_*` и сравнивает их на отложенных парах. Сводка — в конце `eval.log`, ответы — в `eval.jsonl` и `eval.html`. Чтобы получить несколько контрольных точек за одно обучение, задайте `epochs=6 max_steps_per_epoch=250`: адаптер сохраняется каждые 250 шагов.

## Отличия от handoff-dpo

- Базовая модель читается из общей копии `/data/shared/dpo_team/models/Qwen3-8B`, своя копия не нужна.
- Сохраняется только адаптер (`save_adapter_weights_only: True`), а не слитые веса по 16 ГБ на эпоху.
- Пути в `train.sh` абсолютные: Slurm запускает копию скрипта из своей служебной папки, и файлы «рядом со скриптом» там не находятся.
- Пустой блок размышлений добавляет `prepare_data.py`, отдельный шаг `prep_nothink.py` не нужен.
- Гиперпараметры те же: LoRA r=16, alpha=32, lr 5e-5, beta 0.1, 2 эпохи, батч 4 × накопление 4.
