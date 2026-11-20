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

## 2026-11-19 — T1: разбор малых значений таргета (y < 200)
- Запрос: расследовать, почему планка ошибается на y<200 (фолды 2022/2023); проверить три гипотезы (intercept, сдвиг ratio, дискретность проектов); поправку считать только по train-годам; кандидата добавлять только если улучшает малые строки на обоих фолдах и не ухудшает остальные; сабмишн не менять; вывод в DECISIONS; reviewer на утечки, log-step, make ci.
- Модель/агент: Sonnet (анализ, код, тесты), субагент `reviewer` (Opus) — ревью на утечки и корректность выводов.
- Что сделано: функции разбора в `itgorod.eda` (`last_year_rows`, `small_rows`, `elasticity_by_size`, `dept_intercepts`, `predict_elastic`; общая группировка по размеру отдела), скрипт `notebooks/02_t1_small_values.py`, график `docs/figures/t1_small_values.png`. Итог: 2022 — малых проблем нет (6 строк, 2,52%); 2023 — 7 строк, 13,82%, из них 60% APE у одной строки; главная причина — три обвала проектов в 2023 (−56…−70%, y упал на 70–80% при эластичности 1,4–1,6), в train-годах фолдов обвалов почти нет, поправку оценить нельзя. Гипотеза 1 (intercept) не объясняет (знак обратный), 2 (сдвиг ratio) подтверждена только на строках с резким изменением проектов, 3 (дискретность) не обнаружена, но шум ratio растёт с уменьшением отдела. Кандидаты E (общая эластичность) и E2 (α по группам размера) не прошли критерий: малые строки 2023 хуже (+0,57 / +1,19 п.п.). В DECISIONS — запись 11 и альтернативы для решения пользователя, в EXPERIMENTS — две отвергнутые строки. По ревью: тест на утечку в `elasticity_by_size`, тест согласованности p_proj, защита от нулевых проектов, `include_lowest`, пустая группа, смягчены формулировки выводов.
- Файлы: src/itgorod/eda.py, tests/test_eda_small.py (новый), notebooks/02_t1_small_values.py (новый), docs/figures/t1_small_values.png (новый), docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md. `t1.py` и `outputs/task-1.csv` не менялись.
- Проверка: `make ci` → lint ок, 116 passed; мутации (убран фильтр по max_year в `elasticity_by_size`; p_proj без учёта NaN-таргета) → соответствующие тесты красные, код возвращён; `make t1-experiment` → α = 1,038 [0,923; 1,159] без изменений после рефакторинга; `uv run itgorod validate t1 outputs/task-1.csv` → OK 505 rows; `git diff HEAD -- src/itgorod/t1.py` → пусто.
- Что исправил человек: TODO(German)

## 2026-11-19 — T2: разведка данных, предложение схемы валидации
- Запрос: ветка feat/task2, разведка T2 (схема, классы, иерархия, тексты, даты, дубли, код в тексте), замер пиковой памяти, выводы в DECISIONS/AGENTS, скрипт, тесты, reviewer, log-step, make ci; коммит только по команде.
- Модель/агент: Sonnet (код, доки); субагенты explorer/Haiku (схема и шапки, полный прогон скрипта с `/usr/bin/time -v`), reviewer/Opus (ревью).
- Что сделано: `src/itgorod/t2_recon.py` (иерархия, статистика классов, хеши головы и тела, дубли, код в тексте, разбор шапки: тип/дата акта/номер/орган/дата выгрузки, сплит по датам, пересечение органов); `notebooks/03_t2_recon.py` (потоковый проход через `iter_t2_batches`); запись 12 в DECISIONS и T2-раздел AGENTS.md (предложение, не принято). По ревью: строгая регулярка кода, хеш тела без шапки, пересечение органов, поправлены формулировки про память и дубли. Пик RSS ≈ 715–752 МиБ.
- Файлы: src/itgorod/t2_recon.py, notebooks/03_t2_recon.py, tests/test_t2_recon.py, docs/DECISIONS.md, AGENTS.md, docs/AI_LOG.md.
- Проверка: `make ci` → ruff чисто, 133 passed; полный прогон `notebooks/03_t2_recon.py` → exit 0, ~17–27 с. Сабмишнов не создавалось, `validate-submission` не требовался.
- Что исправил человек: TODO(German)

## 2026-11-19 — T2: закрепление схемы валидации, бейзлайны, память
- Запрос: закрепить схему валидации T2 (запись 13), TDD-бейзлайны (LinearSVC, LogisticRegression, ComplementNB; TF-IDF word+char; max_chars 2000/5000; balanced только у LinearSVC), измерить пик памяти, результаты в `outputs/` и `docs/EXPERIMENTS.md`; коммит схемы и кода по команде.
- Модель/агент: Sonnet (код, доки); субагент reviewer/Opus (ревью кода и ревью результатов).
- Что сделано: `t2_cv.py` (нормализация органа, основная и контрольная схемы, срезы OOF, fit только на train-части), `t2_experiment.py` + `itgorod t2 experiment` / `make t2-experiment`; коммит `e37ed02` (код и правила, без результатов). Замер памяти: пик в `fit` LinearSVC (float64-копия), урезание признаков (word 30k, char 50k) дало −3% (4,19 → 4,07 ГиБ), лимит прогона 75% RAM. Полный прогон 7 экспериментов; лучший — LinearSVC без весов, 2000 символов: Macro-F1 0,829 (main) / 0,792 (control). Выводы и ограничения — `docs/DECISIONS.md`, запись 14; 7 строк в `docs/EXPERIMENTS.md`. По ревью: поправлены формулировки (редкие классы, разрыв main/control, худшие классы control, пик RSS по группе), пометки типов, комментарии конфига. Результаты в `outputs/`, не в git; сабмишн не создавался.
- Файлы: src/itgorod/t2_experiment.py, tests/test_t2_experiment.py, configs/t2.yaml, AGENTS.md, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md; (в коммите e37ed02: src/itgorod/t2_cv.py, tests/test_t2_cv.py, cli.py, Makefile).
- Проверка: `make ci` → см. ниже; мутация «fit векторайзера на всех текстах» → тест `test_run_experiments_fit_only_on_train_part` красный, код возвращён; полный `itgorod t2 experiment` → exit 0, пик RSS 4273 МиБ при лимите ≈22,5 ГиБ.
- Что исправил человек: TODO(German)

## 2026-11-19 — T2: страховочный сабмишн (LinearSVC, 2000 символов)
- Запрос: закоммитить результаты сетки; реализовать `run_t2_full` (сначала тест), записать `outputs/task-2.csv`, провалидировать, добавить строку `t2-submit-svc-c2000`, показать распределение классов test против train.
- Модель/агент: Sonnet; субагент reviewer на этом шаге не вызывался.
- Что сделано: коммит результатов сетки; `run_t2_full` (fit векторайзера и модели на всём train, `max_chars`=2000, word 30k + char 50k), CLI `train-predict` без `--smoke` и `make t2`; тесты формата, порядка `row_id`, кодов из train, fit только на train, несовпадения `row_id`. Старый CLI-тест случайно запустил реальный прогон — переписан на заглушку. Запись 15 в DECISIONS, строка в EXPERIMENTS.
- Файлы: src/itgorod/t2.py, src/itgorod/cli.py, tests/test_t2.py, tests/test_cli.py, Makefile, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md; outputs/task-2.csv (не в git).
- Проверка: `uv run itgorod t2 train-predict --force` → exit 0, 114 с, пик RSS 2,86 ГиБ; `uv run itgorod validate t2 outputs/task-2.csv` → OK 10001 rows; независимая проверка pandas: порядок = sample, 58 из 58 классов; `make ci` → см. отчёт.
- Что исправил человек: TODO(German)

## 2026-11-20 — T2: настройка LinearSVC (усечение, голова+хвост, C, вклад шапки)
- Запрос: усилить тест на утечку (шпион на любой fit векторайзера и модели, красный на сломанном коде), записать minor-замечания в known limitations; сетка: max_chars 1000/3000, голова+хвост (1500+500, 2000+1000), C ∈ {0,25; 0,5; 1; 2}, вклад шапки; σ по фолдам, контрольная схема для финалиста; фон через nohup, лимит памяти 75% RAM; `task-2.csv` не менять.
- Модель/агент: Sonnet; субагент reviewer/Opus (ревью кода и результатов).
- Что сделано: `tests/spies.py` + усиленные тесты (5 мутаций fit на чужих данных → красные); режим «голова + хвост» и «без шапки» на уровне Arrow (`compose_text`, эталонный тест); `Experiment` с `tail_chars`/`strip_header`/`C`; `run_experiments` с `scheme_names` и `append`; `t2_tune.py` (правило «прирост > σ», подтверждение control, возобновление) и `itgorod t2 tune` / `make t2-tune`. Итог: усечение остаётся 2000 (ни один выигрыш не превысил σ 0,0086); лучший C=2: main 0,8429 / control 0,8096 против базы 0,8289 / 0,7924, подтверждено контрольной; без шапки main 0,7458 (−0,0971). Оценка победителя оптимистична (выбор по той же CV). Записи 16 в DECISIONS, 8 строк в EXPERIMENTS. Первый запуск упал после этапа 1 из-за моей ошибки в `t2_tune.py`; исправлено, записано.
- Файлы: src/itgorod/data.py, config.py, t2_experiment.py, t2_tune.py, cli.py, Makefile, tests/spies.py, tests/test_t2.py, tests/test_t2_experiment.py, tests/test_t2_data.py, tests/test_t2_tune.py, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md; outputs/t2_experiments.csv, t2_per_class.csv, t2_tune_summary.json, логи (не в git).
- Проверка: `make ci` → см. ниже; `itgorod t2 tune` → exit 0 после перезапуска; `md5sum -c` для `outputs/task-2.csv` → OK (не менялся); пик RSS 3781 МиБ при лимите ≈22,5 ГиБ.
- Что исправил человек: TODO(German)

## 2026-11-20 — T2: C ∈ {4, 8}, абляция шапки по частям, сабмишн C=1
- Запрос: закоммитить настройку; сетка C ∈ {4, 8}, выбор наименьшего C в пределах σ=0,0086 от лучшего, подтверждение контрольной; абляция шапки по частям (только/без типа, органа, номера и дат), обе схемы, падения F1 > 10 п.п.; тест разбора шапки на реальных форматах; заменить `outputs/task-2.csv` на финалиста (fit на всём train, тест на утечку), validate, make ci, записи, reviewer, log-step.
- Модель/агент: Sonnet; субагент reviewer/Opus.
- Что сделано: коммит настройки (`feat: T2 LinearSVC tuning …`); `split_header` / `header_part_text` с тестами на образцах формата реальных шапок; режимы `header_mode` в `Experiment`; `pick_smallest_within_sigma`, `big_drops`, `ablation_grid`, `run_tune_c_and_header` (`itgorod t2 tune2`, `make t2-tune2`). Нашёл и исправил свою ошибку: проверка `n_rows` при возобновлении в прошлый раз не применилась (правка молча не сработала), хотя была записана как сделанная; теперь общий `_runner` + тест. Результаты: выбран C=1 (лучший C=2: 0,8429; C=1: 0,8396 / control 0,8045, подтверждён); шапка: почти весь сигнал в номере и датах (только они: 0,8257 / 0,7946; без них: 0,7525 / 0,7033; тип и орган сверх тела ничего не дают); зонд: дата выгрузки одна даёт accuracy 0,762 (majority 0,348), в test её распределение совпадает с train. `outputs/task-2.csv` заменён на C=1 (`FULL_EXPERIMENT`, тест фиксирует конфиг), 81 строка отличается от страховочного; повторный прогон даёт побайтно тот же файл. Записи 17 в DECISIONS, строки в EXPERIMENTS, правило в AGENTS.md обновлено, зонд — `notebooks/04_t2_header_probe.py`.
- Файлы: src/itgorod/t2.py, t2_recon.py, t2_experiment.py, t2_tune.py, cli.py, Makefile, notebooks/04_t2_header_probe.py, tests/test_t2.py, test_t2_recon.py, test_t2_experiment.py, test_t2_tune.py, AGENTS.md, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md; outputs/ (не в git).
- Проверка: `itgorod t2 tune2` → exit 0; `itgorod t2 train-predict` → exit 0, 114 с, пик RSS 2,6 ГиБ; `itgorod validate t2 outputs/task-2.csv` → OK 10001 rows; повторный прогон → побайтно тот же файл; `make ci` → см. отчёт.
- Что исправил человек: TODO(German)

## 2026-11-20 — T2: проверка стабильности по группам дат выгрузки
- Запрос: закоммитить предыдущий шаг (сделано, `8760d8a`); проверка стабильности (group k-fold по дате выгрузки, C=1, 2000 символов), Macro-F1 и F1 по редким, сравнение с основной схемой, запись в EXPERIMENTS/DECISIONS с формулировкой «оценка включает сигнал порядка сбора, без него ориентир около 0,75», раздел README «Ограничения оценки», reviewer, log-step, make ci; сабмишн не менять.
- Модель/агент: Sonnet; субагент reviewer/Opus.
- Что сделано: `date_group_folds`/`fetch_dates` (хронологические квантильные группы из целых дат), дополнительные схемы в `run_experiments`, `run_date_stability` (`itgorod t2 stability`, `make t2-stability`), `class_group_spread`. Результат: схема вырождена — Macro-F1 ≈ 0,007 у всех трёх вариантов, потому что классы почти целиком привязаны к датам (56 из 58 классов ≥95% строк в одной группе; у 35,0% строк класса нет в train-части). Ориентир «без сигнала порядка сбора» взят из абляции (без номера и дат: 0,7525 main / 0,7033 control), не из этой схемы. Записи 18 в DECISIONS, 3 строки в EXPERIMENTS, раздел README «Ограничения оценки». Ревью нашло критическую ошибку: дозапись схемы `date` стирала колонки main/control у трёх строк `t2_experiments.csv` — исправлено слиянием по id, тест, строки пересчитаны (значения те же); также дубли id в сетке, проверка длины дат. Мои ошибки: правка `t2_tune.py` в предыдущем шаге молча не применилась (повтор: правка summary тоже не сработала при первой попытке), теперь после правок проверяю grep'ом.
- Файлы: src/itgorod/t2_cv.py, t2_experiment.py, t2_tune.py, cli.py, Makefile, tests/test_t2_cv.py, test_t2_experiment.py, test_t2_tune.py, README.md, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md; outputs/ (не в git).
- Проверка: `itgorod t2 stability` → exit 0; `itgorod t2 tune2` (пересчёт трёх строк) → exit 0; `md5sum -c` для `outputs/task-2.csv` → OK (не менялся); `make ci` → см. отчёт; повторное ревью после исправлений не запускалось.
- Что исправил человек: TODO(German)

## 2026-11-20 — T2: подмена даты выгрузки в valid (оценка чувствительности)
- Запрос: reviewer на последние исправления и коммит; затем вариант 1 — обучение как обычно, в valid-текстах дата выгрузки заменяется случайной датой из train-части фолда (seed 42), обе схемы, Macro-F1, F1 по редким, число классов с падением > 10 п.п., сравнение с 0,8396 / 0,8045 и с абляцией (0,7525 / 0,7033); записи, README, тест, что подмена касается только valid; reviewer, log-step, make ci; сабмишн не менять.
- Модель/агент: Sonnet; субагент reviewer/Opus (два раза).
- Что сделано: коммит исправлений (`feat: T2 date-group stability check …`; сначала pre-commit остановил коммит из-за линтера, исправил и закоммитил); `swap_fetch_date` / `swap_valid_dates` / `fetch_date_pool`, поле `perturb` в `Experiment`, `run_swap_date_eval` (`itgorod t2 swapdate`, `make t2-swapdate`); тесты, в том числе «train не меняется, пул дат из train-части, подмена только даты» (две ломающие правки — красные). Результат: Macro-F1 0,6464 main / 0,5895 control (Δ −0,19 / −0,22), редкие 0,6949 / 0,68; классов с падением > 10 п.п. 38 (main) / 43 (control); ниже абляции без номера и дат, потому что это чувствительность к ложной дате, а не оценка «без даты». Записано в EXPERIMENTS, DECISIONS 18, README. «Партиями по классам» оставлено гипотезой. Reviewer без critical; минимальные замечания (формулировка про seed — пара (42, фолд), пул с частотами train, число строк с датой не фиксировалось) исправлены.
- Файлы: src/itgorod/t2_cv.py, t2_experiment.py, t2_tune.py, cli.py, Makefile, tests/spies.py, tests/test_t2_cv.py, test_t2_experiment.py, test_t2_tune.py, README.md, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md; outputs/ (не в git).
- Проверка: `itgorod t2 swapdate` → exit 0; `md5sum -c` для `outputs/task-2.csv` → OK (не менялся); `make ci` → см. отчёт.
- Что исправил человек: TODO(German)

## 2026-11-20 — T2: эмбеддинги как опциональные признаки (ветка feat/embeddings)
- Запрос: ветка feat/embeddings; разведка GPU через explorer; кандидаты энкодера через Hugging Face MCP; замер скорости на 1000 документов; кэш эмбеддингов вне git с ключом в имени; тест на кэш и «энкодер не видит меток»; затем (после установки токена) bge-m3 с лимитом 15 минут, полное кодирование e5-base, OOF LogisticRegression в обеих схемах, бленд с TF-IDF SVC (C=1, 2000), правило σ=0,0086; `task-2.csv` не менять.
- Модель/агент: Sonnet; субагенты explorer/Haiku (GPU и окружение), reviewer/Opus.
- Что сделано: `src/itgorod/embeddings.py` (спецификация и кэш, `HFEncoder`, замер, `embed_corpus`), `src/itgorod/t2_embed_eval.py` (OOF-логиты, бленд, выбор веса, кэш OOF SVC, оркестратор), команды `itgorod t2 embed | oof-svc | embed-eval`, `notebooks/05_t2_embedding_bench.py`, тесты (кэш, ключ, энкодер без меток, fit scaler/LR только на train-части, бленд, пол для отсутствующих классов). Окружение: RTX 2060 SUPER 8 ГБ, torch 2.15 CUDA, ничего не ставилось. Замер e5-base: 206–207 док/с, 810 МиБ, ≈4 мин на 50 тыс. Закодировано train+test в двух режимах (≈8,4 мин). bge-m3 не скачан за 15 минут (≈1,9 из ≈2,3 ГБ) — записано; sbert_large_nlu_ru не пробовался по указанию. Результат: e5+LR Macro-F1 0,62 / 0,57 против 0,84 / 0,80 у TF-IDF; бленд не даёт прироста сверх σ (лучший w=0,2: +0,0018 / +0,0014). Ревью нашло невидимую раньше ошибку (LR на двух классах в train-части возвращал один столбец) — исправлено, тест; не сошедшийся LR (max_iter=300) перепроверен при 1500 — выводы те же. Записи 19 в DECISIONS, строки в EXPERIMENTS.
- Файлы: src/itgorod/embeddings.py, t2_embed_eval.py, cli.py, Makefile, notebooks/05_t2_embedding_bench.py, tests/test_embeddings.py, tests/test_t2_embed_eval.py, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/AI_LOG.md; data/cache/ и outputs/ не в git.
- Проверка: `make ci` → ruff чисто, 216 passed; `md5sum outputs/task-2.csv` до и после совпадает (6022899a640c…); поиск токена по репозиторию, логам и докам — значений нет; `itgorod t2 embed`, `oof-svc`, `embed-eval` → exit 0.
- Что исправил человек: TODO(German)

## 2026-11-20 — Финальная проверка и документация (ветка feat/final)
- Запрос: воспроизведение в чистом клоне, проверка секретов и персональных данных, README, список незаполненных полей, make ci, ревью.
- Модель/агент: Sonnet; субагент reviewer/Opus.
- Что сделано: цели `submit` (t1 + t2, затем `validate-submit`) и `validate-submit` в Makefile; README переписан (цель, дедлайн, воспроизведение, структура, итоги, ограничения оценки, решения, где AI_LOG); правки README по замечаниям reviewer. Модели и сабмишны не менялись.
- Файлы: Makefile, README.md, docs/AI_LOG.md.
- Проверка: чистый клон с origin + копия data/raw, `uv sync --frozen` + `make submit` → 3:18, пик памяти 2,86 ГБ, md5 task-1.csv и task-2.csv совпали с outputs/ (расхождений нет); `gitleaks detect --log-opts=--all` (26 коммитов) → утечек нет; в отслеживаемых файлах и истории нет data/, outputs/, .env, ключей, абсолютных путей /home/, токенов HF/GitHub; `make ci` → ruff чисто, 216 passed.
- Что исправил человек: TODO(German)
