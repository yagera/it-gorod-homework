# ДЗ ML-стажировки (ДИТ Москвы)

Две задачи, сдаются два CSV (`task-1.csv`, `task-2.csv`), дедлайн 22.11.2026, 12:00.

| Задача | Что | Метрика |
|---|---|---|
| T1 | прогноз `yearly_cv_processed` на 2024 по 505 департаментам | MAPE |
| T2 | классификация муниципальных НПА по коду `NNN.NNN.NNN` (58 классов) | Macro-F1 |

Текущее состояние: скелет репозитория. T1 — **плейсхолдер** (значение 2023 по отделу), T2 — **smoke-прогон** на подвыборке. Настоящие модели — следующие этапы, см. `docs/EXPERIMENTS.md`.

## Установка и данные
```
make install          # uv sync (в т.ч. группа dl: torch и др.)
```
Данные не в git: положить выданные файлы в `data/raw/task/` (условия), `data/raw/task1/`, `data/raw/task2/`.

## Команды
```
make t1                  # → outputs/task-1.csv (плейсхолдер)
make t2-prepare-smoke    # усечённые тексты → data/processed
make t2-smoke            # → outputs/task-2.smoke.csv (только проверка формата)
make validate            # формат обоих сабмишнов против sample_submission
make check-time-split    # порядок фолдов T1 и отсутствие будущих лет/таргета 2024 в сырых данных (утечки через признаки не ловит)
make lint | test | ci    # ruff, pytest, оба сразу
```
Все команды — обёртки над CLI `uv run itgorod …`. Тесты работают на синтетике и не требуют `data/`.

## Где что
- `src/itgorod/` — код; `configs/` — YAML-конфиги; `tests/`; `notebooks/` — только EDA.
- `outputs/` — сабмишны (не в git); `data/processed/` — кэш текстов T2 (не в git).
- `AGENTS.md` — правила проекта (метрики, валидация, запреты, Definition of Done); `CLAUDE.md` — модели и агенты.
- `docs/` — `AI_LOG.md` (журнал работы с ИИ), `EXPERIMENTS.md`, `DECISIONS.md`, `superpowers/specs/` (дизайн), `plans/` (план).
