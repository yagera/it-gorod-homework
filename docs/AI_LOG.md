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

## 2026-11-19 — Task 3: проверки утечек по времени и метрики
- Запрос: продолжить с Task 3 плана
- Модель/агент: Sonnet 5.5
- Что сделано: в `validation.py` — `LeakageError`, `split_fold` (expanding window), `assert_no_time_leakage`, `assert_folds_ordered`, `assert_features_for_year` (правило A); `metrics.py` — `mape` (доля), `macro_f1`; фикстура `t1_frame`; тест утечек параметризован по фолдам из `configs/t1.yaml`
- Файлы: src/itgorod/validation.py, src/itgorod/metrics.py, tests/conftest.py, tests/test_time_split.py, tests/test_metrics.py, docs/AI_LOG.md
- Проверка: тесты сначала падали (ошибка импорта), затем `uv run pytest -q` → 35 passed (по одному тесту на каждый фолд); ruff чистый
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 4: чтение T1 и плейсхолдер
- Запрос: продолжить с Task 4 плана
- Модель/агент: Sonnet 5.5
- Что сделано: `data.read_t1` (выбрасывает пустую строку, int-типы), `t1.predict_placeholder` (значение 2023 → последнее известное по отделу → медиана 2023 по `tech_stack` → общая медиана; перед расчётом `assert_features_for_year`), `t1.run_t1` (порядок как в sample, папка вывода создаётся). Это плейсхолдер, не бейзлайн
- Файлы: src/itgorod/data.py, src/itgorod/t1.py, tests/test_t1.py, docs/AI_LOG.md
- Проверка: тесты сначала падали (ошибка импорта), затем `uv run pytest -q` → 43 passed; ruff чистый
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 5: батчевое чтение T2, усечение, кэш
- Запрос: продолжить с Task 5 плана
- Модель/агент: Sonnet 5.5
- Что сделано: в `data.py` — `iter_t2_batches` (усечение `text` на стороне Arrow до перехода в pandas, остановка по `limit`), `cache_path` (в имени `max_chars` и число строк smoke), `build_t2_cache` (test всегда целиком, переиспользование и `force`, папка создаётся), `read_t2_codes` (только колонка `code`); фикстура `t2_parquets`
- Файлы: src/itgorod/data.py, tests/conftest.py, tests/test_t2_data.py, docs/AI_LOG.md
- Проверка: тесты сначала падали (ошибка импорта), затем `uv run pytest -q` → 49 passed; ruff чистый. Реальный train, `limit=5000`, `batch_size=500`, `max_chars=5000`: 5000 строк, пик памяти ≈ 770 МБ, 2.6 с. Наблюдение: parquet из одной row group, для полного train (этап T2) пик памяти надо перемерить
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 6: smoke-пайплайн T2
- Запрос: продолжить с Task 6 плана
- Модель/агент: Sonnet 5.5
- Что сделано: `t2.py` — `build_model` (TfidfVectorizer + LogisticRegression, параметры и `random_state` из конфига), `run_t2_smoke` (кэш → fit на smoke-подвыборке → предсказание всего test → `outputs/task-2.smoke.csv`, порядок как в sample, `code` строкой), `run_t2_full` (NotImplementedError «полный прогон — на этапе T2»); тест воспроизводимости T1 и T2
- Файлы: src/itgorod/t2.py, tests/test_t2.py, tests/test_reproducibility.py, docs/AI_LOG.md
- Проверка: `test_t2.py` сначала падал (ошибка импорта), затем `uv run pytest -q` → 54 passed; ruff чистый. `test_t1_reproducible` прошёл сразу — `run_t1` уже существовал с Task 4, так что RED для него не наблюдался
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 7: CLI, Makefile, прогон на реальных данных
- Запрос: продолжить с Task 7 плана
- Модель/агент: Sonnet 5.5
- Что сделано: `cli.py` (подкоманды `t1 predict`, `t2 prepare|train-predict [--smoke] [--force]`, `validate {t1,t2}`, `check-time-split`; коды возврата 0/1/2), `itgorod.main()` делегирует в CLI, `Makefile` с тонкими обёртками. Реальные данные: T1-плейсхолдер и T2-smoke построены и прошли `validate`
- Файлы: src/itgorod/cli.py, src/itgorod/__init__.py, Makefile, tests/test_cli.py, docs/AI_LOG.md
- Проверка: тесты CLI сначала падали (ошибка импорта); `make ci` → ruff чист, 60 passed. `make t1` + `itgorod validate t1` → OK, 505 строк; `make check-time-split` → оба фолда OK и признаки 2024 OK; `make t2-prepare-smoke` (пик памяти ≈ 0.87 ГБ, 2.4 с) и `make t2-smoke` (пик ≈ 1.5 ГБ, 25 с) + `itgorod validate t2 outputs/task-2.smoke.csv` → OK, 10001 строка. `data/`, `outputs/` в git не попали
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 8: pre-commit и CI
- Запрос: продолжить с Task 8 плана
- Модель/агент: Sonnet 5.5
- Что сделано: `.pre-commit-config.yaml` (ruff-check --fix, ruff-format, gitleaks, check-added-large-files 1000 КБ, end-of-file-fixer, trailing-whitespace, check-yaml; версии — актуальные теги из `git ls-remote`), минимальный `.github/workflows/ci.yml` (job lint-test: `uv sync --frozen --no-group dl`, ruff, pytest; job gitleaks с `fetch-depth: 0`). Решение: `gitleaks-action` оставлен на `@v2` (по спеке), хотя уже есть `v3` — его поведение не проверял
- Файлы: .pre-commit-config.yaml, .github/workflows/ci.yml, docs/AI_LOG.md
- Проверка: `uv run pre-commit run --all-files` → все 7 хуков Passed; имитация CI (`uv sync --frozen --no-group dl` + ruff + pytest) → 60 passed; workflow YAML парсится. Реальный запуск CI на GitHub не проверялся (нужен push — по отдельной команде)
- Что исправил человек: TODO(German)

## 2026-11-19 — Task 9: остальная обвязка агентов
- Запрос: продолжить с Task 9 плана; убрать `disallowedTools` из explorer, запреты оставить текстом контракта, записать остаточный риск, повторить пробу
- Модель/агент: Sonnet 5.5; пробы — субагент explorer (haiku)
- Что сделано: хук ruff format в `settings.json` (deny остался только на `.env*` и `git push`), агенты `reviewer` (opus, Read/Grep/Glob) и `explorer` (haiku, Read/Grep/Glob/Bash), скиллы `validate-submission` и `time-split-check`, раздел про субагентов и скиллы в `CLAUDE.md`, запись об остаточном риске explorer в `docs/DECISIONS.md`
- Файлы: .claude/settings.json, .claude/agents/reviewer.md, .claude/agents/explorer.md, .claude/skills/validate-submission/SKILL.md, .claude/skills/time-split-check/SKILL.md, CLAUDE.md, docs/DECISIONS.md, docs/AI_LOG.md
- Проверка: `jq . .claude/settings.json` валиден; хук на `src/itgorod/seed.py` → exit 0. Первая версия explorer с `disallowedTools` получила только Read/Grep/Glob (без Bash) — убрано. Вторая проба до перезагрузки агентов — Bash всё ещё нет (определение не перечитано). После `/reload-plugins`: `uv run python -I -c "import pyarrow"` выполнен (26.0.1), `rm` отклонён агентом по контракту, файл цел
- Что исправил человек: TODO(German) (перезагрузил агентов; выбрал вариант без `disallowedTools`)

## 2026-11-19 — Task 10: документация, финальная проверка и ревью ветки
- Запрос: продолжить с Task 10 плана
- Модель/агент: Sonnet 5.5; ревью всей ветки — субагент `reviewer` (opus, только чтение)
- Что сделано: `docs/DECISIONS.md` (записи 1–6: без Hydra/MLflow/DVC/Docker, правило A и фолды, единый CLI, валидация T2 — TBD, группа `dl` и минимальный CI, ограничения explorer), `docs/EXPERIMENTS.md` (две строки: `t1-placeholder`, `t2-smoke` — не бейзлайны), `README.md`, `notebooks/.gitkeep`, поправлена устаревшая строка в `AGENTS.md`
- Файлы: docs/DECISIONS.md, docs/EXPERIMENTS.md, README.md, notebooks/.gitkeep, AGENTS.md, docs/AI_LOG.md
- Проверка: `make lint` чист, `make test` → 60 passed, `make t1`, `make t2-prepare-smoke`, `make t2-smoke`, `make validate` (OK 505 / OK 10001 строк), `make check-time-split` (оба фолда и признаки 2024 OK), `pre-commit run --all-files` → 7 хуков Passed, имитация CI (`--frozen --no-group dl`) → 60 passed; `data/` и `outputs/` игнорируются. Ревью ветки: Critical и Important не найдено; 11 замечаний уровня Minor отложены (список — в ответе пользователю), правок кода по ним не вносилось
- Что исправил человек: TODO(German)

## 2026-11-19 — Исправления по ревью ветки (замечания 9, 7, 1, 8)
- Запрос: исправить замечания 9, 7, 1 и 8; тесты сначала; остальные (2–6, 10, 11) — в DECISIONS.md блоком «Известные ограничения скелета», без правок
- Модель/агент: Sonnet 5.5
- Что сделано: (7) `itgorod validate` при отсутствующем файле сабмишна/sample/train печатает `ERROR: file not found: <путь>` и возвращает 2 (1 — только нарушения формата); (9) тест на `code`, прочитанный как число; (1) скилл `time-split-check` и README честно описывают, что проверяет `check-time-split` и что нет (утечки через признаки не ловит); (8) скилл `validate-submission`: число строк берётся из sample, а не зашито; блок «Известные ограничения скелета» (замечания 2–6, 10, 11) в `docs/DECISIONS.md`
- Файлы: src/itgorod/cli.py, tests/test_cli.py, tests/test_validation.py, .claude/skills/time-split-check/SKILL.md, .claude/skills/validate-submission/SKILL.md, README.md, docs/DECISIONS.md, docs/AI_LOG.md
- Проверка: два теста для (7) сначала падали (`FileNotFoundError`), после правки `make ci` → 63 passed, ruff чист. Тест для (9) прошёл сразу — код уже отвергал числовой `code`; чтобы проверить, что тест что-то охраняет, временно убрал `isinstance`-проверку → тест упал с `TypeError`, код возвращён. `make validate` → OK 505 / OK 10001; `itgorod validate t1 outputs/nope.csv` → ERROR, exit 2
- Что исправил человек: TODO(German)

## 2026-11-19 — T1: EDA (ветка feat/task1)
- Запрос: ветка feat/task1, EDA T1 как notebooks/01_t1_eda.py (ячейки `# %%`), логика в src/itgorod, графики в docs/figures/, 6 гипотез, 5–7 выводов в DECISIONS.md.
- Модель/агент: Sonnet (код, EDA), субагент `reviewer` (Opus) — ревью; правки по ревью внесены (тесты на утечку valid-года, сравнение моделей на общих строках, недостающие печати, противоречие п.5/п.6).
- Что сделано: `src/itgorod/eda.py` (качество, таргет, ratio, тождества, дрейф, tech_stack, диагностика по временным фолдам), `read_t1_raw`, скрипт EDA с 5 графиками, запись 7 в DECISIONS (7 выводов). Главное: target ≈ ratio_отдела × active_projects_count (медианный CV ratio внутри отдела 2,3%); диагностический MAPE 2,8–3,5% против 4,4–6,2% у лага таргета; есть точные тождества budget=1,2·staff·salary и turnover=(hired+fired)/staff; сильный дрейф только у avg_dev_salary (+8,6%).
- Файлы: src/itgorod/eda.py, src/itgorod/data.py, notebooks/01_t1_eda.py, tests/test_eda.py, docs/DECISIONS.md, docs/figures/t1_*.png (5 шт.), docs/AI_LOG.md
- Проверка: `uv run ruff format . && uv run ruff check .` → ок; `uv run pytest -q` → 70 passed; `uv run python notebooks/01_t1_eda.py` → выполняется, 5 графиков сохранены; `make check-time-split` → оба фолда и признаки 2024 OK.
- Что исправил человек: TODO(German)

## 2026-11-19 — T1: подходы A + D, проверка α, условный B
- Запрос: brainstorming подходов поверх ratio-планки → выбор A + D; план (writing-plans), TDD; раннер экспериментов на обоих фолдах; проверка α; B (Ridge на log ratio) — только по условию; C зафиксировать как отклонённый; ревью на утечки; итоговая таблица.
- Модель/агент: Opus (brainstorming, план, код, исполнение плана), субагент `reviewer` (Opus) — финальное ревью на утечки.
- Что сделано: план `docs/superpowers/plans/2026-11-19-t1-ratio-smoothing.md`; в `itgorod.t1` — модель A (`predict_ratio`: смесь r_last, среднего и затухающего тренда, исключение выбросов > 15%, fallback-цепочка), D (`fit_multiplier`: взвешенная медиана y/p по внутреннему бэктесту), B (`predict_ridge`); раннер `itgorod.t1_experiment` (`make t1-experiment`: общие строки, срез y<200, парный бутстрэп, сетка w × phi); `projects_elasticity` в `eda.py`. Итог: ни A, ни A+D, ни B не обходят планку (2,84% / 3,06%); α = 1,04 [0,92; 1,16]. Условие для B выполнилось (A+D не ниже планки), B реализован и тоже не лучше. В первой версии B был признак, который на обучении почти константа → переполнение exp; признак убран, ẑ ограничен диапазоном обучения. По ревью: усилен мутационный тест утечки (2024, все колонки будущего, точное сравнение — проверено внедрением двух утечек), уточнена запись 9 (планка лучше выбранной точки A по сумме фолдов; порог выбросов вредит на 2023).
- Файлы: src/itgorod/t1.py, src/itgorod/t1_experiment.py (новый), src/itgorod/eda.py, src/itgorod/config.py, src/itgorod/cli.py, configs/t1.yaml, Makefile, tests/test_t1_ratio.py (новый), tests/test_t1_experiment.py (новый), tests/test_config.py, tests/test_eda.py, docs/DECISIONS.md (записи 8–9), docs/EXPERIMENTS.md (4 строки), docs/superpowers/plans/2026-11-19-t1-ratio-smoothing.md, docs/AI_LOG.md
- Проверка: `make lint` → ok; `make test` → 98 passed; `make check-time-split` → оба фолда и признаки 2024 OK; `make t1-experiment` → `outputs/experiments.csv`, `outputs/t1_grid.csv`; мутационная проверка теста утечки: внедрённые утечки → 2 failed, после отката → passed.
- Что исправил человек: TODO(German)

## 2026-11-19 — T1: правки по ревью и сабмишн = планка
- Запрос: исправить три замечания ревьюера (тест множителя c, ошибка при пустой истории в `predict_ratio`, устаревший docstring), остальное — в known limitations; перевести `run_t1` на планку (ratio отдела за 2023 × проекты test), записать выбор в DECISIONS со ссылкой на `outputs/experiments.csv`, пометить замену плейсхолдера в EXPERIMENTS; `validate-submission`, `make ci`, проверить, что `.superpowers/` и `outputs/` вне git.
- Модель/агент: Sonnet.
- Что сделано: (а) тавтологичный тест множителя заменён `test_multiplier_scales_prediction_and_reduces_error` + `test_fold_predictions_applies_multiplier`; на мутантах (множитель всегда 1; раннер игнорирует c) тесты красные, на исходном коде зелёные; (б) `predict_ratio` (а значит и `predict_ridge`) при пустой истории → `ValueError("history is empty…")`, тест был красным до правки; (в) docstring `t1.py` переписан. `predict_placeholder` удалён, вместо него `predict_baseline` (проверка правила A, отказ при нефинитном/неположительном прогнозе), `run_t1` падает, если для id из sample нет прогноза (тест красный до правки); фикстура `t1_frame` получила `active_projects_count`. DECISIONS: запись 10 (почему планка, что проверено, чего не знаем) и обновлённые known limitations; EXPERIMENTS: плейсхолдер помечен заменённым, добавлена строка `t1-submit-ratio-last`. `outputs/task-1.csv` пересобран.
- Файлы: src/itgorod/t1.py, tests/conftest.py, tests/test_t1.py, tests/test_t1_ratio.py, tests/test_t1_experiment.py, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md (outputs/task-1.csv — вне git)
- Проверка: `make t1` → written, 506 строк (шапка + 505); независимый пересчёт ratio 2023 × проекты 2024 из сырых CSV → максимальное расхождение 0,0; `uv run itgorod validate t1 outputs/task-1.csv` → OK 505 rows; `make validate` → OK для T1 и T2-smoke; `make check-time-split` → OK; `make ci` → lint ок, 103 passed; `git check-ignore` → `.superpowers/`, `outputs/` игнорируются.
- Что исправил человек: TODO(German)
