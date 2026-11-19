# Скелет репозитория ДЗ ML-стажировки (ДИТ Москвы) — дизайн

Дата: 2026-11-19. Дедлайн сдачи: 22.11.2026, 12:00.

## 1. Зачем

Два задания ML-стажировки: **T1** — прогноз `yearly_cv_processed` на 2024 по 505 департаментам (регрессия, MAPE); **T2** — классификация муниципальных НПА по коду `NNN.NNN.NNN` (58 классов, Macro-F1). Сдаются только `task-1.csv` и `task-2.csv`; EDA и код живут в репозитории. Проверяющие отдельно интересуются работой с ИИ, поэтому процесс (журнал шагов, решения, эксперименты) — часть результата.

Скелет нужен, чтобы с первого дня: (а) существовал рабочий путь «сырые данные → валидный CSV» для обеих задач; (б) типовые ошибки (формат сабмишна, ведущие нули в коде, утечки по времени, невоспроизводимость) ловились автоматически — тестами, хуками, CI и скиллами; (в) работа с ИИ-агентами шла по явным правилам и протоколировалась.

## 2. Цели / не-цели / анти-цели

**Цели**
- uv + src-layout, пакет `itgorod`, единый CLI `itgorod`, YAML-конфиги, Makefile как точка входа.
- Плейсхолдер T1 и smoke-пайплайн T2, оба дают CSV, проходящий `validate-submission`.
- pytest на синтетике (CI без данных): формат сабмишна, отсутствие утечек будущих лет в T1, воспроизводимость (seed=42).
- ruff, pre-commit с gitleaks, GitHub Actions (ruff, pytest, gitleaks).
- AGENTS.md, CLAUDE.md, `.claude/` (settings, 2 субагента, 3 скилла), `docs/` (AI_LOG, EXPERIMENTS, DECISIONS).

**Не-цели**
- Настоящие бейзлайны и выбор подхода для T1 (после EDA).
- Полный прогон T2 по всему train (~2 ГБ) и выбор схемы валидации T2 (на этапе T2).
- Сам EDA (только папка `notebooks/` под него).

**Анти-цели** (провал, даже если «работает»)
- Плейсхолдер T1 или smoke T2 записаны в EXPERIMENTS.md как бейзлайн/результат.
- `code` где-либо читается/пишется не как строка (потеря ведущих нулей).
- Тесты или CI требуют реальных данных.
- Данные или `.env` попадают в git / в контекст агента.
- В AGENTS.md заранее зафиксирована схема валидации T2.

## 3. Ограничения
- Без Hydra, MLflow, DVC, Docker (обоснование — в DECISIONS.md, см. §6.6).
- Зависимости — только уже объявленные в `pyproject.toml`; новые не добавлять без нужды. **Решение пользователя:** тяжёлые пакеты (torch, transformers, sentence-transformers, catboost, lightgbm) переносятся из `[project.dependencies]` в группу `[dependency-groups] dl`; `[tool.uv] default-groups = ["dev", "dl"]` (локально `uv sync` ставит всё как прежде); `uv.lock` пересобирается. Скелет использует только pandas, pyarrow, scikit-learn, pyyaml, numpy.
- **Git — только по команде пользователя:** работа в отдельной ветке, без коммитов и push без явного «да».
- `data/` и `outputs/` в `.gitignore` (уже есть; дубли строк в `.gitignore` убрать).
- `.claude/settings.json` уже содержит `enabledPlugins` (superpowers) — дополнять, не перезаписывать.
- Существующая точка входа `[project.scripts] itgorod = "itgorod:main"` сохраняется; `main()` делегирует в CLI.

## 4. Подходы

1. **Один CLI `itgorod` с подкомандами + библиотека** — *выбран.* Проверки — обычные функции в `itgorod/validation.py`, их вызывают pytest, скиллы, CI и Makefile; одна реализация на всех.
2. Отдельные скрипты в `scripts/` — быстрее, но проверки расползаются копиями, хуже тестируются.
3. Ноутбуки + nbconvert — плохо тестируются и ложатся на ruff/CI. Отброшен; ноутбуки только для EDA.

## 5. Зафиксированные решения пользователя

- **Правило утечки T1 (A):** для года Y можно использовать все признаки года Y и любые данные лет < Y (включая лаги таргета). Нельзя — таргет года Y и любые данные лет > Y. Валидация T1 — только по времени (напр. train 2020–2022 → valid 2023).
- **T1 сейчас — плейсхолдер:** значение 2023 по отделу; фолбэк при пропуске — последнее известное значение отдела, иначе медиана 2023 по `tech_stack`. В EXPERIMENTS.md — как «плейсхолдер».
- **T2 сейчас — smoke:** чтение батчами → усечение → кэш → TF-IDF + LogReg → сабмишн на подвыборке train в несколько тысяч строк. Test читается целиком (10 001 строка), но батчами с усечением при чтении. Сабмишн помечен как smoke.
- **Валидация T2** в AGENTS.md не фиксируется; там пометка «TBD: решается после проверки, разбит ли test по времени относительно train (дата в шапке документа)».
- Предполагаем те же 58 классов в test.
- Модели: Opus — план и ревью, Sonnet — код, Haiku — разведка и документация.

## 6. Дизайн

### 6.1 Структура

```
AGENTS.md  CLAUDE.md  README.md  Makefile  pyproject.toml  .pre-commit-config.yaml  .gitignore
configs/        t1.yaml  t2.yaml
src/itgorod/    __init__.py  cli.py  config.py  seed.py  data.py  validation.py  metrics.py
                t1.py  t2.py
tests/          conftest.py (синтетические фикстуры)  test_validation.py  test_time_split.py
                test_reproducibility.py  test_t2_data.py  test_cli.py
notebooks/      .gitkeep   (EDA)
docs/           AI_LOG.md  EXPERIMENTS.md  DECISIONS.md  superpowers/specs/
.claude/        settings.json  agents/{reviewer,explorer}.md
                skills/{validate-submission,time-split-check,log-step}/SKILL.md
.github/workflows/ci.yml
data/raw/...  data/processed/   outputs/        (не в git)
```

### 6.2 Модули
- `config.py` — dataclass'ы `T1Config`, `T2Config` (+ вложенные), `load_config(path) -> dataclass` через `yaml.safe_load`; неизвестный ключ → ошибка. Пути в конфиге — относительно корня репо.
- `seed.py` — `set_seed(seed)`: `random`, `numpy`; `seed` из конфига, по умолчанию 42, пробрасывается в `random_state` моделей и в сэмплинг.
- `data.py`
  - `read_t1(path)`: CSV, выбрасывает полностью пустые строки (в train есть одна), `department_id`/`year` → int.
  - `iter_t2_batches(path, columns, batch_size, max_chars)`: `pyarrow.ParquetFile.iter_batches`, текст усекается **внутри батча** до `max_chars` (голова документа — там шапка); `code` остаётся строкой.
  - `build_t2_cache(split, cfg, smoke)`: пишет усечённые данные в `data/processed/t2_{split}_c{max_chars}[_smoke{n}].parquet`; при наличии файла — переиспользует (флаг `--force` пересобирает). Smoke для train — первые `n` строк (сейчас `n=5000`, задаётся в конфиге; проверено: покрывают 56 из 58 классов — для smoke достаточно). Parquet состоит из одной row group — стриминг через `iter_batches(batch_size=…)` строитель обеспечивает сам и останавливает чтение после `n` строк.
  - `read_t2_codes(path)` — только колонка `code` (для списка допустимых классов).
  - Все чтения CSV/parquet для T2 — с `dtype=str` / строковым типом для `code`.
- `validation.py` (ядро для тестов, скиллов, CI)
  - `validate_t1_submission(sub, sample)`: колонки ровно `department_id,yearly_cv_processed`; 505 строк (= len(sample)); порядок `department_id` как в sample; значения конечные и > 0; без дублей.
  - `validate_t2_submission(sub, sample, allowed_codes)`: колонки ровно `row_id,code`; 10001 строка; `row_id` совпадает с sample по порядку; `code` — строка, `fullmatch(r"\d{3}\.\d{3}\.\d{3}")`, ∈ `allowed_codes`; без пропусков. Читает CSV с `dtype=str`.
  - `assert_no_time_leakage(train_df, valid_df, year_col="year")`: `max(train.year) < min(valid.year)`.
  - `assert_features_for_year(features_df, target_year, target_col)`: нет строк с `year > target_year`; таргет года `target_year` не присутствует среди признаков (правило A).
  - Все функции бросают `SubmissionError`/`LeakageError` с перечнем всех нарушений, а не только первого.
- `metrics.py` — `mape`, `macro_f1` (тонкие обёртки над sklearn, единое место).
- `t1.py` — `predict_placeholder(train, test) -> DataFrame` (правило из §5), сборка признаков проходит через `assert_features_for_year`; `run(cfg)` пишет `outputs/task-1.csv` в порядке sample.
- `t2.py` — `train_smoke(cfg)`: кэш → `TfidfVectorizer` + `LogisticRegression` (параметры и `random_state` из конфига) → предсказание test → `outputs/task-2.smoke.csv`; `code` пишется строкой.
- `cli.py` (argparse):
  - `itgorod t1 predict --config configs/t1.yaml`
  - `itgorod t2 prepare --config configs/t2.yaml [--smoke] [--force]`
  - `itgorod t2 train-predict --config configs/t2.yaml --smoke` (без `--smoke` пока падает с понятным сообщением «полный прогон — на этапе T2»)
  - `itgorod validate {t1,t2} <csv>` — пути к sample и train берутся из конфига; код возврата ≠ 0 при ошибке
  - `itgorod check-time-split --config configs/t1.yaml` — проверяет **каждый фолд** из `split.folds` на реальных данных T1

### 6.3 Конфиги (пример полей)
- `t1.yaml`: `seed`, `paths.{train,test,sample,output}`, `split.folds` — **список фолдов** (правка пользователя): `[{train_max_year: 2021, valid_year: 2022}, {train_max_year: 2022, valid_year: 2023}]` (expanding window: train = все годы ≤ `train_max_year`), `target_year: 2024`, `placeholder.fallback: tech_stack_median`.
- `t2.yaml`: `seed`, `paths.{train,test,sample,processed_dir,output}`, `read.{batch_size, max_chars}`, `smoke.n_train_rows: 5000`, `tfidf.{…}`, `logreg.{…}`. Валидационная схема T2 — отсутствует до решения.

### 6.4 Makefile (тонкие обёртки)
`install` (uv sync), `lint` (ruff check + format --check), `format`, `test`, `t1`, `t2-prepare-smoke`, `t2-smoke`, `validate` (обе), `check-time-split`, `ci` (= lint + test). Все через `uv run`.

### 6.5 Тесты (только синтетика, без `data/`)
- Синтетические фикстуры в `conftest.py`: мини-T1 (≈10 отделов × 2020–2023 + test 2024, с пропусками таргета) и мини-T2 (parquet в `tmp_path` с кодами вроде `010.000.000`, `020.150.030`, длинными текстами).
- `test_validation.py`: корректный сабмишн проходит; ломаются: неверные колонки, число строк, порядок id, ≤0/NaN в T1, код без ведущих нулей (`10.0.0`), код вне train, код как число после `read_csv` без `dtype=str`.
- `test_time_split.py`: каждый фолд из `split.folds` (≤2021→2022, ≤2022→2023) проходит — тест параметризован по фолдам, читаемым из `configs/t1.yaml`; фолды строго возрастают по времени; пересечение лет и признаки `year > Y` падают; плейсхолдер T1 не использует данные 2024-таргета/будущих лет.
- `test_reproducibility.py`: два прогона T2-smoke на синтетике с seed=42 дают идентичные предсказания; T1-плейсхолдер детерминирован.
- `test_t2_data.py`: батчевое чтение усекает до `max_chars`, кэш создаётся и переиспользуется, `code` остаётся str.
- `test_cli.py`: `itgorod validate` возвращает 0/≠0.

### 6.6 docs/
- `DECISIONS.md` — короткие записи (дата, решение, причина, альтернативы). Начальные: (1) без Hydra/MLflow/DVC/Docker — два небольших задания, один исполнитель, 3 дня, сдача — 2 CSV; YAML+dataclass, EXPERIMENTS.md, `uv.lock` и seed закрывают конфиги, трекинг и воспроизводимость; данные выданы локально и не версионируются; инфраструктура съела бы время без выигрыша в метрике. (2) Правило утечки T1 = A. (3) Единый CLI. (4) Валидация T2 — TBD. (6) Deny на rm/mv/git commit/git push — защита от explorer частичная, остаточный риск принят. (5) Тяжёлые ML-пакеты — в группе `dl`, CI ставит `--no-group dl` (иначе несколько ГБ CUDA-пакетов на каждый push); CI минимальный и может быть отключён, если станет дорогим.
- `EXPERIMENTS.md` — таблица: дата, задача, id/название, тип (`placeholder`/`smoke`/`baseline`/`experiment`), конфиг/коммит, валидация, метрика, комментарий. Первые две строки — T1 placeholder и T2 smoke (метрика «—, не оценивается»).
- `AI_LOG.md` — шаблон записи (см. log-step) + первая запись про этот шаг.

### 6.7 AGENTS.md / CLAUDE.md
- `AGENTS.md`: цель и сдача (имена файлов, дедлайн), метрики, структура репо, команды Makefile; правила валидации (T1 — только по времени, правило A; T2 — TBD); жёсткие запреты: не читать `.env*`, не коммитить `data/`, `outputs/`, секреты; `code` только как str; Definition of Done: `make lint` и `make test` зелёные, `validate-submission` проходит для затронутых сабмишнов, `time-split-check` для изменений T1, запись в EXPERIMENTS.md (если был прогон) и в AI_LOG.md (log-step), решение — в DECISIONS.md, если принималось.
- `CLAUDE.md`: `@AGENTS.md` + таблица «модель → роль» (Opus — план и ревью, Sonnet — код, Haiku — разведка и документация) + какие субагенты/скиллы использовать когда.

### 6.8 `.claude/`
- `settings.json`: сохраняет `enabledPlugins`; `permissions.deny` — **только** `Read(./.env)`, `Read(./.env.*)`, `Read(**/.env*)` и `Bash(git push:*)` (правка пользователя); хук `PostToolUse` на `Edit|Write|MultiEdit` → берёт `tool_input.file_path` через `jq`, для `*.py` запускает `uv run ruff format <file>`. Запреты `rm`, `mv`, `git commit` — только в описании `explorer`; коммит основная сессия делает лишь по команде «коммить» на чекпойнте.
- `agents/reviewer.md`: `model: opus`, tools: Read, Grep, Glob (только чтение). Ищет утечки (правило A, таргет-лаги, fit на valid/test), ошибки схемы валидации, нарушения формата сабмишна и `code` как не-str; выдаёт список находок с файлом/строкой.
- `agents/explorer.md`: `model: haiku`, tools: Read, Grep, Glob, Bash. Разведка данных: метаданные parquet, `read_table(columns=…)`, `iter_batches`; запрещено грузить T2 text целиком и что-либо изменять/создавать вне scratch.
- `skills/validate-submission/SKILL.md`: вызывает `make validate` / `uv run itgorod validate {t1,t2} <csv>`, интерпретирует ошибки. Требования — §6.2.
- `skills/time-split-check/SKILL.md`: напоминает правило A, запускает `make check-time-split` и `pytest tests/test_time_split.py`, плюс чеклист ручной проверки кода фичей (лаги, агрегаты по отделу/стеку не включают valid/test-годы).
- `skills/log-step/SKILL.md`: дописывает в конец `docs/AI_LOG.md` запись: дата, шаг, запрос, модель/агент, что сделано, изменённые файлы, проверка (команды и результат), «Что исправил человек: TODO(German)».

### 6.9 Качество
- `pyproject.toml`: секция `[tool.ruff]` (line-length 100, target py311, правила E,F,I,B,UP), `[tool.pytest.ini_options]` (`testpaths=["tests"]`).
- `.pre-commit-config.yaml`: ruff (check --fix, format), gitleaks, `check-added-large-files`, `end-of-file-fixer`, `trailing-whitespace`.
- `.github/workflows/ci.yml` — **минимальный, один workflow** (решение пользователя): на push/PR; job `lint-test` (setup-uv, `uv sync --frozen --no-group dl`, ruff check, ruff format --check, pytest) — без torch и без реальных данных; job `gitleaks` (`gitleaks/gitleaks-action@v2`, `fetch-depth: 0`). Если CI начнёт съедать много времени — отключаем; основная проверка локально: `make test` и `pre-commit`.

## 7. Проверка готовности скелета
1. `make lint`, `make test` — зелёные (локально, без данных тоже).
2. `make t1 && uv run itgorod validate t1 outputs/task-1.csv` → OK.
3. `make t2-prepare-smoke && make t2-smoke && uv run itgorod validate t2 outputs/task-2.smoke.csv` → OK; в `data/processed/` есть кэш.
4. `make check-time-split` → OK.
5. `pre-commit run --all-files` → OK (включая gitleaks).
6. Записи в EXPERIMENTS.md (placeholder, smoke), DECISIONS.md, AI_LOG.md.
7. `uv sync --frozen --no-group dl && uv run pytest` в чистом окружении проходит (имитация CI).
8. После push по команде пользователя — CI на GitHub зелёный.

## 8. Решения автора дизайна (можно оспорить)
- Фолбэк плейсхолдера T1: последнее известное значение отдела → медиана 2023 по `tech_stack`.
- Smoke T2 — первые 5000 строк train, сабмишн в `outputs/task-2.smoke.csv` (не `task-2.csv`, чтобы не спутать со сдачей).
- Усечение — голова документа (там шапка с типом акта и органом).
- Валидаторы собирают все нарушения сразу, а не падают на первом.
- Хук ruff format — `PostToolUse` на `Edit|Write|MultiEdit` через `jq`, только для `*.py`.
- reviewer — только Read/Grep/Glob; explorer — плюс Bash (для pyarrow). **Правка пользователя (финальная):** запреты `rm`, `mv`, `git commit`, `git push` — в описании explorer; в `settings.json` из них остаётся только `git push`. После создания агента проверить применение через `/agents`; если запреты не применились — остаточный риск в DECISIONS.md. Ограничение частичное: Bash нельзя закрыть полностью.

## 9. Оставлено строителю
Точные гиперпараметры TF-IDF/LogReg для smoke, `max_chars` и `batch_size` (разумные дефолты, напр. 5000 символов и 500 строк), текст README, формулировки AGENTS.md в рамках §6.7, точные имена фикстур и тестов, разбиение коммитов.
