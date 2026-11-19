# AI_LOG

Журнал работы с ИИ (требование ДЗ: показать, как идёт совместная работа). Записи добавляются
скиллом `log-step`. Поле «Что исправил человек» заполняет German.

Шаблон — в `.claude/skills/log-step/SKILL.md`.

Записи до Task 0b добавлены вручную задним числом, дальше — скиллом log-step.

## 2026-11-18 — Разведка условий и данных
- Запрос: прочитать homework.pdf, посмотреть данные T1/T2 (parquet только метаданные), вернуть сводку
- Модель/агент: Opus 5.5
- Что сделано: прочитан PDF (9 слайдов), профиль T1 (505 отделов × 2020–2023, пропуски, бейзлайн «2023=2022» MAPE 6.2%), метаданные T2 (39 996 / 10 001 строк, 58 классов, иерархия кода, дисбаланс 1161:1), список рисков и вопросов
- Файлы: —
- Проверка: read-only скрипты pandas/pyarrow → цифры в сводке
- Что исправил человек: TODO(German)

## 2026-11-19 — Брейншторм, спека и план скелета
- Запрос: скелет репозитория (uv, src-layout, CLI, тесты, CI, агенты, скиллы, docs)
- Модель/агент: Opus 5.5; builder-check — субагент general-purpose
- Что сделано: решения (правило утечки A, один CLI, группа `dl`, минимальный CI, фолды T1 списком), спека и план на 11 задач
- Файлы: docs/superpowers/specs/2026-11-19-repo-scaffold-design.md, docs/plans/2026-11-19-repo-scaffold.md
- Проверка: builder-check вернул один важный вопрос (тяжёлые зависимости в CI) → решено
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 0: ветка, зависимости, конфиг инструментов
- Запрос: выполнить Task 0 плана
- Модель/агент: Sonnet 5.5
- Что сделано: тяжёлые пакеты в группу `dl`, `default-groups`, ruff/pytest в pyproject, чистка .gitignore, пересборка uv.lock
- Файлы: pyproject.toml, uv.lock, .gitignore, tests/test_smoke_import.py
- Проверка: `uv sync --frozen --no-group dl && uv run pytest -q` → 1 passed; `import torch` → ModuleNotFoundError (как и должно быть в CI)
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 0b: AGENTS.md, CLAUDE.md, log-step, настройки
- Запрос: вынести AGENTS.md, CLAUDE.md и log-step в начало; в settings.json только deny на .env* и git push; отключить атрибуцию в коммитах/PR
- Модель/агент: Sonnet 5.5; ключ `attribution` уточнён субагентом claude-code-guide по документации
- Что сделано: AGENTS.md (цели, метрики, правило A, T2 — TBD, запреты, DoD), CLAUDE.md (импорт + модели/роли), скилл log-step, AI_LOG.md, settings.json: deny `.env*` и `git push`, `attribution` пустой + `sessionUrl: false`
- Файлы: AGENTS.md, CLAUDE.md, .claude/settings.json, .claude/skills/log-step/SKILL.md, docs/AI_LOG.md
- Проверка: `jq . .claude/settings.json` валиден, `enabledPlugins` на месте; `uv run pytest -q` → 1 passed
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 1: конфиги и seed
- Запрос: продолжить с Task 1 плана
- Модель/агент: Sonnet 5.5
- Что сделано: `config.py` (dataclass'ы T1/T2, `Fold`, загрузка YAML, отказ на неизвестный ключ во всех секциях, включая элементы `split.folds[i]`; ключи `tfidf` и `logreg` не проверяются — они передаются в sklearn; пути от корня репо), `seed.py`, `configs/t1.yaml` (`split.folds` списком), `configs/t2.yaml` (без схемы валидации — TBD)
- Файлы: src/itgorod/config.py, src/itgorod/seed.py, configs/t1.yaml, configs/t2.yaml, tests/test_config.py, docs/AI_LOG.md
- Проверка: сначала тест падал (ошибка импорта), затем `uv run pytest tests/test_config.py -v` → 4 passed; по замечанию человека добавлен тест на лишний ключ в фолде (упал с TypeError), после правки `_folds` с `_check_keys` → 5 passed; `ruff check` и `ruff format --check` чистые
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 2: валидаторы сабмишнов
- Запрос: продолжить с Task 2 плана
- Модель/агент: Sonnet 5.5
- Что сделано: `validation.py` — `SubmissionError` (накапливает все нарушения), `read_submission_csv` (`dtype=str`), `validate_t1_submission`, `validate_t2_submission` (формат `NNN.NNN.NNN`, только коды train, порядок `row_id`); фикстуры в `tests/conftest.py`
- Файлы: src/itgorod/validation.py, tests/conftest.py, tests/test_validation.py, docs/AI_LOG.md
- Проверка: тесты сначала падали (ошибка импорта), затем `uv run pytest -q` → 24 passed; ruff чистый (ruff поймал длинную строку и форматирование — исправлено); по замечанию человека `CODE_RE` переведён с `\d` на `[0-9]` (строгий ASCII): добавлен тест на юникодные цифры (код в `allowed_codes`, иначе тест не падал бы на старой регулярке) — RED на `\d`, затем 26 passed
- Что исправил человек: TODO(German)
