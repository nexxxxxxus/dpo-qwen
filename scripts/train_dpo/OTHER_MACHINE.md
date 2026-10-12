# Перебор гиперпараметров DPO на другой машине

В архиве всё, чтобы продолжить перебор гиперпараметров на машине с такими же картами и драйвером, как на нашем сервере: код, данные и эта инструкция. Базовой модели и окружения внутри нет — они ставятся на месте, шаги 2 и 3.

## Что обучаем

LoRA-адаптер для Qwen3-8B методом DPO: модель учится выделять жирным главное в коротком ответе. Перебираем скорость обучения (lr) и beta, каждую комбинацию проверяем на отложенных вопросах.

## Что нужно от машины

- Linux, карта NVIDIA с памятью от 32 ГБ (считали на A100 80 ГБ; обучение занимает 28 ГБ);
- драйвер NVIDIA 570.x (CUDA 12.x) — окружение собрано под CUDA 12.9;
- около 45 ГБ на диске: окружение 13 ГБ, модель 16 ГБ, остальное — кэши и результаты;
- conda и доступ в интернет (PyPI, GitHub, Hugging Face).

## 1. Распаковать

Архив рассчитан на распаковку в домашнюю папку: все пути в скриптах по умолчанию идут от неё.

```bash
tar -xzf dpo_highlight_bundle.tar.gz -C ~
```

Появятся:

- `~/dpo-qwen/` — код;
- `~/data/dpo-qwen/data/tuning_two_kinds/` — `train.jsonl` (17 625 пар), `validation.jsonl` (1001 вопрос), `test.jsonl` (2001 вопрос), `gold.jsonl` (эталонные ответы SberQuAD для проверки);
- `~/README.md` — эта инструкция.

## 2. Поставить окружение

```bash
cd ~/dpo-qwen
scripts/train_dpo/setup_env.sh
```

Около 15 минут и 13 ГБ. Скрипт создаёт conda-окружение с Python 3.12 в `~/dpo-qwen/.venv`, ставит точные версии пакетов из `requirements.lock.txt`, правит один файл в torchtune (без правки обучение падает с `NameError: TypeVar`) и проверяет, что всё импортируется. В конце должно быть `GPU visible to torch: True` и `environment is ready`.

Если conda лежит не в `PATH` и не в `/opt/conda/bin/conda`, укажите её: `CONDA=/путь/к/conda scripts/train_dpo/setup_env.sh`.

Сообщения `Failed to load ... _C_cutlass_90a` и `_C_mxfp8` при импорте — от необязательных модулей torchao, на работу не влияют. `pip check` ругается на версию `datasets` для `trl` — это известно, `trl` в обучении не участвует.

## 3. Скачать веса

```bash
tmux new -s model 'cd ~/dpo-qwen && scripts/train_dpo/download_model.sh'
```

16 ГБ с Hugging Face (`Qwen/Qwen3-8B`) в `~/data/dpo-qwen/models/Qwen3-8B`. Токен не нужен. Другое место: `MODEL=/путь scripts/train_dpo/download_model.sh` — тогда перед запуском обучения выполните `export MODEL=/путь`.

## 4. Запустить перебор

Две комбинации уже посчитаны на нашем сервере (lr 1e-5 с beta 0.05 и 0.1), осталось семь. Каждая — 250 шагов обучения (около 25 минут на A100, плюс 5 минут на сохранение) и проверка на 1001 вопросе (около 10 минут).

**Без Slurm** — запуски идут друг за другом на одной карте, всё вместе около пяти часов:

```bash
tmux new -s tune
cd ~/dpo-qwen
GPU=0 LRS="1e-5" BETAS="0.3" CHECKPOINTS=1 scripts/train_dpo/tune_local.sh
GPU=0 LRS="2e-5 5e-5" BETAS="0.05 0.1 0.3" CHECKPOINTS=1 scripts/train_dpo/tune_local.sh
```

`GPU` — номер карты. Чтобы занять две карты, откройте второе окно tmux и разделите комбинации между ними (например, `LRS="2e-5"` на карте 0 и `LRS="5e-5"` на карте 1). Скрипт можно запускать повторно: уже проверенные комбинации он пропускает.

**Со Slurm** — задачи уходят в очередь, по одной карте на каждую:

```bash
cd ~/dpo-qwen
export MODEL=~/data/dpo-qwen/models/Qwen3-8B
LRS="1e-5" BETAS="0.3" CHECKPOINTS=1 TRAIN_TIME=1:00:00 EVAL_TIME=0:30:00 scripts/train_dpo/tune.sh
LRS="2e-5 5e-5" CHECKPOINTS=1 TRAIN_TIME=1:00:00 EVAL_TIME=0:30:00 scripts/train_dpo/tune.sh
```

Ход одного запуска: `tail -f ~/data/dpo-qwen/outputs/<папка запуска>/slurm.log` (шаги обучения) и `eval.log` (проверка).

## 5. Посмотреть результаты

```bash
cd ~/dpo-qwen/scripts/train_dpo
LD_LIBRARY_PATH=~/dpo-qwen/.venv/lib ~/dpo-qwen/.venv/bin/python tune_table.py --runs "*tune-*"
```

Печатает таблицу по всем проверенным запускам и сохраняет её в `~/data/dpo-qwen/outputs/tune_table.csv`. Столбцы:

| Столбец | Что значит | Чего хотим |
|---|---|---|
| `1-2 highlights` | ответов с одним-двумя выделениями | ближе к 1001 |
| `highlight as in reference` | выделено то же, что в образце | ближе к 1001 |
| `same words as base` | текст ответа тот же, что у базовой модели | как можно больше |
| `has reference answer` | в ответе есть эталон SberQuAD | не ниже строки `base` |
| `median length` | длина ответа в символах | как у `base` |

Для сравнения — две комбинации, посчитанные у нас (`base` — модель без адаптера):

| lr | beta | 1-2 highlights | highlight as in reference | same words as base | has reference answer | median length |
|---|---|---|---|---|---|---|
| base | | 0 | 0 | 1001 | 942 | 79 |
| 1e-5 | 0.05 | 927 | 947 | 462 | 878 | 83 |
| 1e-5 | 0.1 | 958 | 948 | 537 | 889 | 85 |

## 6. Что прислать обратно

Из каждой папки запуска `~/data/dpo-qwen/outputs/<дата>-tune-lr<lr>-beta<beta>/`:

- `eval_validation.jsonl` и `eval_validation.html` — ответы моделей (около 5 МБ);
- `torchtune_config.yaml` — с какими параметрами шло обучение;
- `logs/` — метрики обучения по шагам;
- `epoch_0/adapter_model.safetensors` и `adapter_config.json` — сам адаптер, 87 МБ; нужен только для комбинации, которую выберем.

И общий `~/data/dpo-qwen/outputs/tune_table.csv`.

## Чего не делать

- **Не проверять на `test.jsonl`.** Гиперпараметры выбираются по `validation.jsonl`; `test.jsonl` используется один раз, для итоговой оценки выбранного варианта. Скрипты из шага 4 его не трогают.
- **Не менять `train.jsonl`.** Он собран так, чтобы не пересекаться с `validation` и `test` ни по вопросам, ни по абзацам. Проверить можно так: `.venv/bin/python scripts/train_dpo/check_split.py ~/data/dpo-qwen/data/tuning_two_kinds/{train,validation,test}.jsonl` — везде должны быть нули.

## Если что-то пошло не так

- `CXXABI_1.3.15 not found` — Python из окружения запущен без `LD_LIBRARY_PATH=~/dpo-qwen/.venv/lib`. Скрипты из шагов 2–4 задают его сами.
- `NameError: TypeVar` при обучении — не применилась правка torchtune; запустите `setup_env.sh` ещё раз.
- Обучение упало при сохранении адаптера — проверьте место на диске.
- В `eval.log` сервер vLLM не поднялся за 15 минут — посмотрите конец лога; чаще всего занят порт (`PORT=30031 ...` перед командой) или не найдена модель (`export MODEL=...`).

Подробности об устройстве обучения и данных — в `~/dpo-qwen/scripts/train_dpo/README.md`.
