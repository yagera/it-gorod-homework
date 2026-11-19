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
