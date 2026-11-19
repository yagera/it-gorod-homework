# Repo Scaffold (ДЗ ML-стажировки ДИТ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Скелет репозитория с единым CLI `itgorod`, плейсхолдером T1 и smoke-пайплайном T2, дающими CSV, которые проходят `validate-submission`, плюс тесты на синтетике, линтеры, CI и обвязка для ИИ-агентов.

**Architecture:** Библиотека `src/itgorod/` (config, seed, data, validation, metrics, t1, t2) + тонкий argparse-CLI `itgorod`; Makefile — обёртки над `uv run itgorod …`. Проверки (формат сабмишна, утечки по времени) — обычные функции в `validation.py`, их используют pytest, скиллы, CI и Makefile.

**Tech Stack:** Python 3.11, uv, pandas, pyarrow, scikit-learn, pyyaml, numpy, pytest, ruff, pre-commit, gitleaks, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-19-repo-scaffold-design.md`

## Global Constraints

- Без Hydra, MLflow, DVC, Docker.
- Новых зависимостей не добавлять. torch, transformers, sentence-transformers, catboost, lightgbm → `[dependency-groups] dl`; `[tool.uv] default-groups = ["dev", "dl"]`; CI: `uv sync --frozen --no-group dl`.
- Скелет импортирует только pandas, pyarrow, scikit-learn, pyyaml, numpy (код в `src/` не импортирует пакеты группы `dl`).
- `code` (T2) везде — `str` формата `\d{3}\.\d{3}\.\d{3}`; CSV читать `dtype=str`.
- Тесты и CI не читают `data/`; только синтетика в `tmp_path`.
- seed = 42 по умолчанию, из конфига.
- Правило утечки T1 (A): для года Y — все признаки года Y и любые данные лет < Y; запрещены таргет года Y и любые данные лет > Y.
- Валидация T2 не фиксируется (TBD) ни в AGENTS.md, ни в конфиге.
- Smoke T2 пишет в `outputs/task-2.smoke.csv`; T1 — `outputs/task-1.csv`.
- **Git:** работа в ветке `scaffold`. На каждом «Checkpoint» — сначала `log-step` (запись в `docs/AI_LOG.md`), затем предложить сообщение коммита и ждать. Коммит делаю я, но только когда пользователь напишет «коммить» на этом чекпойнте. `git push` — только по отдельной команде; в `settings.json` он в deny.
- **`settings.json`:** только deny на чтение `.env*` и на `git push` (+ хук ruff format). Запреты на `rm`, `mv`, `git commit` — только в описании `explorer`.
- ruff: line-length 100, target py311, правила `E,F,I,B,UP`.

## Review Focus

1. Сабмишн, записанный с индексом (`to_csv(index=True)`) → лишняя колонка `Unnamed: 0`; валидатор должен отклонять, писатели — писать `index=False`. Тест — Task 2.
2. Отдел T1 без единого известного таргета → плейсхолдер берёт медиану 2023 по `tech_stack`, ни одного NaN/≤0 на выходе. Тест — Task 4.
3. Смена `max_chars` или `n_train_rows` не должна переиспользовать старый кэш → они входят в имя файла. Тест — Task 5.
4. Smoke-модель видит не все 58 классов → предсказания всё равно ⊆ кодов train и проходят валидатор (allowed_codes берутся из полного столбца `code`). Тест — Task 6.
5. `outputs/` или `data/processed/` не существует → создаются автоматически. Тест — Task 5 и Task 7.

---

## File Structure

| Файл | Ответственность |
|---|---|
| `pyproject.toml` | группы зависимостей, ruff, pytest |
| `configs/t1.yaml`, `configs/t2.yaml` | параметры задач |
| `src/itgorod/__init__.py` | `main()` → `cli.main()` |
| `src/itgorod/config.py` | dataclass-конфиги, `load_t1_config`, `load_t2_config` |
| `src/itgorod/seed.py` | `set_seed` |
| `src/itgorod/validation.py` | валидаторы сабмишнов и утечек, исключения |
| `src/itgorod/metrics.py` | `mape`, `macro_f1` |
| `src/itgorod/data.py` | чтение T1, батчевое чтение/усечение/кэш T2 |
| `src/itgorod/t1.py` | плейсхолдер T1 |
| `src/itgorod/t2.py` | smoke TF-IDF+LogReg |
| `src/itgorod/cli.py` | argparse-подкоманды |
| `tests/conftest.py` + `tests/test_*.py` | синтетика и тесты |
| `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | точки входа и качество |
| `AGENTS.md`, `CLAUDE.md`, `.claude/**` | обвязка агентов (AGENTS/CLAUDE/log-step — Task 0b, остальное — Task 9) |
| `docs/{AI_LOG,EXPERIMENTS,DECISIONS}.md`, `README.md`, `notebooks/.gitkeep` | документация |

---

### Task 0: Ветка, зависимости, базовая конфигурация инструментов

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `.gitignore`
- Create: `tests/test_smoke_import.py`

- [ ] **Step 1:** `git checkout -b scaffold`.
- [ ] **Step 2:** В `pyproject.toml` перенести torch, transformers, sentence-transformers, catboost, lightgbm (с текущими версиями) из `[project.dependencies]` в `[dependency-groups] dl`; добавить `[tool.uv] default-groups = ["dev", "dl"]`, `[tool.ruff]` (line-length=100, target-version="py311", `[tool.ruff.lint] select=["E","F","I","B","UP"]`), `[tool.pytest.ini_options] testpaths=["tests"]`.
- [ ] **Step 3:** `.gitignore` — убрать дубли; итог: `data/`, `outputs/`, `.env`, `.env.*`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`.
- [ ] **Step 4:** `uv lock` → Expected: lock обновлён без ошибок.
- [ ] **Step 5:** Тест `tests/test_smoke_import.py::test_import` — `import itgorod; assert callable(itgorod.main)`.
- [ ] **Step 6:** Имитация CI: `uv sync --frozen --no-group dl && uv run pytest -q` → `1 passed`; `uv run python -c "import torch"` → ModuleNotFoundError. Затем `uv sync` (вернуть полное окружение).
- [ ] **Step 7: Checkpoint** — `log-step`; `chore: split heavy deps into dl group, tool config` (после чекпойнта 0 `log-step` появится в Task 0b; запись о Task 0 добавить ретроспективно).

### Task 0b: AGENTS.md, CLAUDE.md, скилл log-step, AI_LOG.md (вынесено в начало)

**Files:**
- Create: `AGENTS.md`, `CLAUDE.md`, `.claude/skills/log-step/SKILL.md`, `docs/AI_LOG.md`
- Modify: `.claude/settings.json` (сохранить `enabledPlugins`)

- [ ] **Step 1: `AGENTS.md`** по спеке §6.7: цель/сдача (`task-1.csv`, `task-2.csv`, Яндекс.Диск, 22.11 12:00), метрики (MAPE, Macro-F1), структура, команды Makefile (помечены «появятся в Task 7»); валидация: T1 — только по времени, фолды из `configs/t1.yaml`, правило A дословно; T2 — «TBD: после проверки, разбит ли test по времени относительно train (дата в шапке документа)»; запреты: не читать `.env*`, не коммитить `data/`, `outputs/`, секреты, `code` только str, не грузить тексты T2 целиком; git: коммит только по команде «коммить» на чекпойнте, push только по отдельной команде; Definition of Done (спека §6.7).
- [ ] **Step 2: `CLAUDE.md`**: первая строка `@AGENTS.md`; таблица «Opus — план и ревью / Sonnet — код / Haiku — разведка и документация»; после каждого чекпойнта вызывать `log-step`.
- [ ] **Step 3: `.claude/settings.json`** (минимум): `permissions.deny`: `Read(./.env)`, `Read(./.env.*)`, `Read(**/.env*)`, `Bash(git push:*)`. Хук ruff format добавляется в Task 9.
- [ ] **Step 4: Скилл `log-step`** (frontmatter `name`, `description` с триггером «после каждого чекпойнта/завершённого шага»): дописать в конец `docs/AI_LOG.md` блок
  ```
  ## YYYY-MM-DD — <шаг>
  - Запрос: …
  - Модель/агент: …
  - Что сделано: …
  - Файлы: …
  - Проверка: <команды> → <результат>
  - Что исправил человек: TODO(German)
  ```
- [ ] **Step 5: `docs/AI_LOG.md`** — заголовок, шаблон и ретроспективные записи: разведка данных, брейншторм/спека, план (с `TODO(German)`).
- [ ] **Step 6:** `jq . .claude/settings.json` валиден; `enabledPlugins` на месте.
- [ ] **Step 7: Checkpoint** — `log-step`; `docs: agents.md, claude.md, log-step skill, ai log`.

### Task 1: Конфиги и seed

**Files:**
- Create: `src/itgorod/config.py`, `src/itgorod/seed.py`, `configs/t1.yaml`, `configs/t2.yaml`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Fold(train_max_year: int, valid_year: int)`
  - `T1Config(seed: int, paths: T1Paths, folds: list[Fold], target_year: int, fallback: str)`; `T1Paths(train, test, sample, output: Path)`
  - `T2Config(seed: int, paths: T2Paths, batch_size: int, max_chars: int, smoke_n_train_rows: int, tfidf: dict, logreg: dict)`; `T2Paths(train, test, sample, processed_dir, output, smoke_output: Path)`
  - `load_t1_config(path: str | Path) -> T1Config`, `load_t2_config(path) -> T2Config`; относительные пути разрешаются от корня репо (`REPO_ROOT = Path(__file__).resolve().parents[2]`); неизвестный ключ → `ValueError` с именем ключа.
  - `set_seed(seed: int) -> None` (random, numpy).

YAML (значения из спеки):
```yaml
# configs/t1.yaml
seed: 42
paths: {train: data/raw/task1/df_ITs_train.csv, test: data/raw/task1/df_ITs_test.csv,
        sample: data/raw/task1/sample_submission.csv, output: outputs/task-1.csv}
split:
  folds:
    - {train_max_year: 2021, valid_year: 2022}
    - {train_max_year: 2022, valid_year: 2023}
target_year: 2024
placeholder: {fallback: tech_stack_median}
```
```yaml
# configs/t2.yaml
seed: 42
paths: {train: data/raw/task2/internship_train.parquet, test: data/raw/task2/internship_test.parquet,
        sample: data/raw/task2/sample_submission.csv, processed_dir: data/processed,
        output: outputs/task-2.csv, smoke_output: outputs/task-2.smoke.csv}
read: {batch_size: 500, max_chars: 5000}
smoke: {n_train_rows: 5000}
tfidf: {max_features: 50000, ngram_range: [1, 2], sublinear_tf: true}
logreg: {C: 4.0, max_iter: 1000, class_weight: balanced}
# validation: TBD — решается на этапе T2 (см. DECISIONS.md)
```

- [ ] **Step 1: Failing tests** `tests/test_config.py`:
  - `test_t1_config_loads`: `cfg.seed == 42`, `cfg.folds == [Fold(2021, 2022), Fold(2022, 2023)]`, `cfg.target_year == 2024`, `cfg.paths.train.is_absolute()`.
  - `test_t2_config_loads`: `cfg.max_chars == 5000`, `cfg.smoke_n_train_rows == 5000`, `cfg.paths.smoke_output.name == "task-2.smoke.csv"`.
  - `test_unknown_key_rejected`: YAML в `tmp_path` с лишним `foo: 1` → `pytest.raises(ValueError, match="foo")`.
  - `test_set_seed_deterministic`: `set_seed(42); a=np.random.rand(3); set_seed(42); assert (a==np.random.rand(3)).all()`.
- [ ] **Step 2:** `uv run pytest tests/test_config.py -v` → FAIL (ImportError).
- [ ] **Step 3:** Реализовать модули по Interfaces.
- [ ] **Step 4:** `uv run pytest tests/test_config.py -v` → 4 passed.
- [ ] **Step 5: Checkpoint** — `log-step`; `feat: yaml configs with dataclasses and seed`.

### Task 2: Валидаторы сабмишнов

**Files:**
- Create: `src/itgorod/validation.py`, `tests/conftest.py`, `tests/test_validation.py`

**Interfaces:**
- Produces:
  - `class SubmissionError(ValueError)` — `errors: list[str]`, сообщение = все ошибки через `\n`.
  - `CODE_RE = re.compile(r"\d{3}\.\d{3}\.\d{3}")`
  - `validate_t1_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None`
  - `validate_t2_submission(sub: pd.DataFrame, sample: pd.DataFrame, allowed_codes: set[str]) -> None`
  - `read_submission_csv(path) -> pd.DataFrame` — `pd.read_csv(path, dtype=str)`.
  - Фикстуры `conftest.py`: `t1_sample` (10 id 1000000..1000009), `t2_sample` (row_id "1".."20", code "000.000.000"), `t2_codes = {"010.000.000", "020.150.030", "290.000.000"}`.

Правила (спека §6.2): T1 — колонки ровно `["department_id","yearly_cv_processed"]`, `len == len(sample)`, `department_id` по порядку == sample, значения парсятся во float, конечны, > 0, без дублей id. T2 — колонки ровно `["row_id","code"]`, `len == len(sample)`, `row_id` по порядку == sample (сравнение строк), нет пустых, каждый `code` — `CODE_RE.fullmatch` и ∈ `allowed_codes`. Ошибки накапливаются, бросаются одной `SubmissionError`.

- [ ] **Step 1: Failing tests** (все sub — DataFrame из str, как после `read_submission_csv`):
  - `test_t1_valid_passes`
  - `test_t1_rejects`: параметризация — лишняя колонка `Unnamed: 0` (Review Focus 1); на строку меньше; переставленный порядок id; значение `"0"`; `"-5"`; `""`/NaN; `"inf"` → `SubmissionError`.
  - `test_t1_collects_all_errors`: sub с неверным порядком И нулём → `len(exc.errors) >= 2`.
  - `test_t2_valid_passes`
  - `test_t2_rejects`: `"10.0.0"`; `"10.000.000"` (потерян ведущий ноль); `"999.000.000"` (вне train); `"010000000"`; лишняя колонка; неверный `row_id` порядок; пустой code → `SubmissionError`.
  - `test_t2_roundtrip_keeps_zeros`: DataFrame с `"010.000.000"` → `to_csv(index=False)` → `read_submission_csv` → проходит валидатор, `code` побайтно равен исходному.
- [ ] **Step 2:** `uv run pytest tests/test_validation.py -v` → FAIL.
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** → все PASS.
- [ ] **Step 5: Checkpoint** — `log-step`; `feat: submission validators`.

### Task 3: Проверки утечек по времени и метрики

**Files:**
- Modify: `src/itgorod/validation.py`
- Create: `src/itgorod/metrics.py`, `tests/test_time_split.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Fold`, `load_t1_config` (Task 1).
- Produces:
  - `class LeakageError(ValueError)` (как `SubmissionError`, с `errors`).
  - `split_fold(df: pd.DataFrame, fold: Fold, year_col="year") -> tuple[pd.DataFrame, pd.DataFrame]` — train = `year <= train_max_year`, valid = `year == valid_year`.
  - `assert_no_time_leakage(train: pd.DataFrame, valid: pd.DataFrame, year_col="year") -> None` — `max(train.year) < min(valid.year)`.
  - `assert_folds_ordered(folds: list[Fold]) -> None` — у каждого `train_max_year < valid_year`; `valid_year` строго возрастают.
  - `assert_features_for_year(features: pd.DataFrame, target_year: int, target_col: str, year_col="year") -> None` — нет строк `year > target_year`; нет непустого `target_col` в строках `year == target_year`.
  - `mape(y_true, y_pred) -> float` (доля, не %; `sklearn.metrics.mean_absolute_percentage_error`), `macro_f1(y_true, y_pred) -> float`.
  - Фикстура `t1_frame` в `conftest.py`: 10 отделов × 2020–2023, `tech_stack` из 3 значений, `yearly_cv_processed` > 0, у одного отдела все таргеты NaN, у другого NaN в 2023.

- [ ] **Step 1: Failing tests** `tests/test_time_split.py`:
  - `test_fold_has_no_leakage[fold]` — параметризация по `load_t1_config("configs/t1.yaml").folds` (ids `"<=2021->2022"`, `"<=2022->2023"`); на `t1_frame`: `split_fold` + `assert_no_time_leakage` проходят, `valid.year.unique() == [fold.valid_year]`, `train.year.max() == fold.train_max_year`.
  - `test_folds_ordered` — конфиг проходит; `[Fold(2022,2023), Fold(2021,2022)]` и `[Fold(2022,2022)]` → `LeakageError`.
  - `test_overlap_detected` — train с годом 2023 против valid 2023 → `LeakageError`.
  - `test_features_future_year_rejected` — строка 2025 при `target_year=2024` → `LeakageError`.
  - `test_features_target_of_year_rejected` — непустой таргет в строке 2024 → `LeakageError`; признаки 2024 без таргета + история ≤2023 → проходит (правило A).
  - `tests/test_metrics.py`: `mape([100,200],[110,180]) == pytest.approx(0.1)`; `macro_f1(["a","b"],["a","a"]) == pytest.approx(1/3)`.
- [ ] **Step 2:** → FAIL. **Step 3:** реализовать. **Step 4:** → PASS (по тесту на каждый фолд).
- [ ] **Step 5: Checkpoint** — `log-step`; `feat: time-split leakage checks and metrics`.

### Task 4: Чтение T1 и плейсхолдер

**Files:**
- Create: `src/itgorod/data.py` (T1-часть), `src/itgorod/t1.py`, `tests/test_t1.py`

**Interfaces:**
- Consumes: `T1Config`, `assert_features_for_year`, `validate_t1_submission`.
- Produces:
  - `read_t1(path) -> pd.DataFrame` — `read_csv`, `dropna(how="all")`, `department_id`, `year` → int.
  - `predict_placeholder(train: pd.DataFrame, test: pd.DataFrame, target_year: int) -> pd.DataFrame` — колонки `department_id, yearly_cv_processed`, порядок как в `test`. Перед расчётом: `assert_features_for_year(pd.concat([train, test]), target_year, "yearly_cv_processed")`; использует только строки `year < target_year`. Значение = таргет `target_year-1` → иначе последний известный таргет отдела → иначе медиана таргета `target_year-1` по `tech_stack` → иначе общая медиана `target_year-1`.
  - `run_t1(cfg: T1Config) -> Path` — читает данные, предсказывает, переупорядочивает по sample, создаёт папку, пишет `to_csv(index=False)`, возвращает путь.

- [ ] **Step 1: Failing tests** `tests/test_t1.py` (на `t1_frame` + синтетическом test 2024 без таргета):
  - `test_read_t1_drops_empty_row` — CSV с полностью пустой последней строкой → строк на 1 меньше, `year` dtype int.
  - `test_placeholder_uses_last_year` — для отдела с таргетом 2023 = 777 → 777.
  - `test_placeholder_fallback_last_known` — отдел с NaN в 2023 и 500 в 2022 → 500.
  - `test_placeholder_fallback_stack_median` — отдел без таргетов → медиана 2023 его `tech_stack` (Review Focus 2); на выходе нет NaN и все > 0.
  - `test_placeholder_rejects_future` — добавить строку 2025 в train → `LeakageError`.
  - `test_placeholder_deterministic` — два вызова → `pd.testing.assert_frame_equal`.
  - `test_run_t1_writes_valid_csv` — `T1Config` с путями в `tmp_path` (папки output нет) → файл создан, `validate_t1_submission(read_submission_csv(out), sample)` проходит.
- [ ] **Step 2:** FAIL → **Step 3:** реализовать → **Step 4:** PASS.
- [ ] **Step 5: Checkpoint** — `log-step`; `feat: t1 placeholder (last-year value with fallbacks)`.

### Task 5: Батчевое чтение T2, усечение, кэш

**Files:**
- Modify: `src/itgorod/data.py`
- Create: `tests/test_t2_data.py`

**Interfaces:**
- Consumes: `T2Config`.
- Produces:
  - `iter_t2_batches(path, columns: list[str], batch_size: int, max_chars: int, limit: int | None = None) -> Iterator[pd.DataFrame]` — `pq.ParquetFile(path).iter_batches(batch_size=…, columns=…)`; в каждом батче `text = text.str.slice(0, max_chars)`; `code` → str; останавливается после `limit` строк (последний батч обрезается).
  - `cache_path(cfg: T2Config, split: Literal["train","test"], smoke: bool) -> Path` — `processed_dir / f"t2_{split}_c{max_chars}{'_smoke' + str(n) if smoke and split=='train' else ''}.parquet"`.
  - `build_t2_cache(cfg, split, smoke: bool, force: bool = False) -> Path` — если файл есть и не `force` → вернуть путь; иначе собрать из батчей (`limit = smoke_n_train_rows` только для train при smoke; test всегда целиком), `mkdir(parents=True)`, записать parquet.
  - `read_t2_codes(path) -> set[str]` — `pq.read_table(path, columns=["code"])`.
  - Фикстура `t2_parquets(tmp_path)` в `conftest.py`: train (60 строк; коды из `t2_codes`, каждый ≥ 10 раз; тексты длиной 50–20 000 символов с ключевым словом класса в начале) и test (20 строк, `row_id` 1..20, int64).

- [ ] **Step 1: Failing tests:**
  - `test_batches_truncate` — все `len(text) <= max_chars` (max_chars=100), суммарно 60 строк, `code` dtype object/str, значения с ведущими нулями.
  - `test_batches_limit` — `limit=25, batch_size=10` → ровно 25 строк.
  - `test_cache_created_and_reused` — первый вызов создаёт (папки нет — Review Focus 5); второй не перезаписывает (mtime не изменился); `force=True` перезаписывает.
  - `test_cache_path_depends_on_params` — разные `max_chars` или `n_train_rows` → разные пути (Review Focus 3); test-путь не зависит от smoke.
  - `test_read_t2_codes` → `== t2_codes`.
- [ ] **Step 2:** FAIL → **Step 3:** реализовать → **Step 4:** PASS.
- [ ] **Step 5: Checkpoint** — `log-step`; `feat: t2 batched reading, truncation and cache`.

### Task 6: Smoke-пайплайн T2

**Files:**
- Create: `src/itgorod/t2.py`, `tests/test_t2.py`, `tests/test_reproducibility.py`

**Interfaces:**
- Consumes: `build_t2_cache`, `read_t2_codes`, `set_seed`, `validate_t2_submission`, `read_submission_csv`.
- Produces:
  - `build_model(cfg: T2Config) -> sklearn.pipeline.Pipeline` — `TfidfVectorizer(**tfidf, ngram_range=tuple(...))` + `LogisticRegression(**logreg, random_state=cfg.seed)`.
  - `run_t2_smoke(cfg: T2Config, force: bool = False) -> Path` — `set_seed`, кэши train(smoke)/test, fit, predict, DataFrame `row_id` (str из кэша test), `code` (str), порядок как в sample, `mkdir`, `to_csv(index=False)` в `cfg.paths.smoke_output`.
  - `run_t2_full(cfg)` — `raise NotImplementedError("полный прогон — на этапе T2")`.

- [ ] **Step 1: Failing tests:**
  - `test_t2_smoke_writes_valid_csv` — конфиг на `t2_parquets`, `smoke_n_train_rows=40`, `max_features=1000` → файл проходит `validate_t2_submission(..., allowed_codes=read_t2_codes(train))`; все коды с ведущими нулями.
  - `test_t2_smoke_subset_missing_class` — train-подвыборка без одного класса → вывод ⊆ `t2_codes`, валидатор проходит (Review Focus 4).
  - `test_t2_full_not_implemented` → `NotImplementedError`.
  - `tests/test_reproducibility.py::test_t2_smoke_reproducible` — два прогона (`force=True`) с seed 42 → CSV побайтно равны. `::test_t1_reproducible` — два `run_t1` → файлы равны.
- [ ] **Step 2:** FAIL → **Step 3:** реализовать → **Step 4:** PASS.
- [ ] **Step 5: Checkpoint** — `log-step`; `feat: t2 smoke tfidf+logreg pipeline`.

### Task 7: CLI, Makefile, прогон на реальных данных

**Files:**
- Create: `src/itgorod/cli.py`, `Makefile`, `tests/test_cli.py`
- Modify: `src/itgorod/__init__.py`

**Interfaces:**
- Consumes: всё выше.
- Produces: `cli.main(argv: list[str] | None = None) -> int`; `itgorod.main()` → `sys.exit(cli.main())`. Подкоманды:
  - `t1 predict --config PATH` → `run_t1`
  - `t2 prepare --config PATH [--smoke] [--force]` → `build_t2_cache` для train и test
  - `t2 train-predict --config PATH [--smoke] [--force]` → `run_t2_smoke`; без `--smoke` → сообщение «полный прогон — на этапе T2», код 2
  - `validate {t1,t2} CSV [--config PATH]` (по умолчанию `configs/t1.yaml`/`configs/t2.yaml`) → печатает `OK` / список ошибок; код 0/1. Для t2 `allowed_codes = read_t2_codes(cfg.paths.train)`.
  - `check-time-split --config PATH` → для каждого фолда `split_fold` + `assert_no_time_leakage` на `read_t1(train)`, плюс `assert_folds_ordered` и `assert_features_for_year(train ∪ test, target_year, …)`; печатает строку на фолд; код 0/1.

Makefile (все через `uv run`): `install` (`uv sync`), `lint` (`ruff check .` + `ruff format --check .`), `format`, `test` (`pytest -q`), `t1`, `t2-prepare-smoke`, `t2-smoke`, `validate` (t1 по `outputs/task-1.csv` + t2 по `outputs/task-2.smoke.csv`), `check-time-split`, `ci` (= lint test). `.PHONY` для всех.

- [ ] **Step 1: Failing tests** `tests/test_cli.py` (через `cli.main([...])` и конфиги в `tmp_path`):
  - `test_validate_ok_returns_0` / `test_validate_bad_returns_1` (T1 с нулём).
  - `test_t2_train_predict_without_smoke_returns_2`.
  - `test_check_time_split_ok` на синтетике → 0; с train-строкой 2025 → 1.
- [ ] **Step 2:** FAIL → **Step 3:** реализовать → **Step 4:** PASS; `uv run pytest -q` — всё зелёное.
- [ ] **Step 5: Реальные данные:**
  - `make t1 && uv run itgorod validate t1 outputs/task-1.csv` → `OK`, 505 строк.
  - `make check-time-split` → 2 фолда OK.
  - `make t2-prepare-smoke` → в `data/processed/` файлы `t2_train_c5000_smoke5000.parquet`, `t2_test_c5000.parquet`; пиковая память — без полной загрузки текста (смотреть `/usr/bin/time -v`, ожидаемо < 2 ГБ).
  - `make t2-smoke && uv run itgorod validate t2 outputs/task-2.smoke.csv` → `OK`, 10001 строка.
- [ ] **Step 6: Checkpoint** — `log-step`; `feat: itgorod cli and makefile`.

### Task 8: pre-commit и CI

**Files:**
- Create: `.pre-commit-config.yaml`, `.github/workflows/ci.yml`

- [ ] **Step 1:** `.pre-commit-config.yaml`: `astral-sh/ruff-pre-commit` (`ruff-check --fix`, `ruff-format`), `gitleaks/gitleaks` (hook `gitleaks`), `pre-commit/pre-commit-hooks` (`check-added-large-files --maxkb=1000`, `end-of-file-fixer`, `trailing-whitespace`, `check-yaml`). Версии — актуальные теги на момент реализации.
- [ ] **Step 2:** `.github/workflows/ci.yml`, один workflow `ci` на `push`/`pull_request`: job `lint-test` (checkout, `astral-sh/setup-uv` с Python 3.11, `uv sync --frozen --no-group dl`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`); job `gitleaks` (checkout `fetch-depth: 0`, `gitleaks/gitleaks-action@v2` с `GITHUB_TOKEN`).
- [ ] **Step 3:** `uv run pre-commit install && uv run pre-commit run --all-files` → все хуки Passed (после автофиксов — повторный прогон чистый).
- [ ] **Step 4:** Имитация CI: `uv sync --frozen --no-group dl && uv run ruff check . && uv run ruff format --check . && uv run pytest -q` → зелёное; затем `uv sync`.
- [ ] **Step 5: Checkpoint** — `log-step`; `ci: pre-commit with gitleaks and minimal github workflow`.

### Task 9: Остальная обвязка агентов

**Files:**
- Create: `.claude/agents/reviewer.md`, `.claude/agents/explorer.md`, `.claude/skills/validate-submission/SKILL.md`, `.claude/skills/time-split-check/SKILL.md`
- Modify: `.claude/settings.json`, `CLAUDE.md`

- [ ] **Step 1: Хук** в `.claude/settings.json`: `hooks.PostToolUse`, matcher `Edit|Write|MultiEdit`, command
  `f=$(jq -r '.tool_input.file_path // empty'); case "$f" in *.py) uv run ruff format "$f" >/dev/null 2>&1 ;; esac; exit 0`. Deny-список не расширять (остаётся `.env*` и `git push`).
- [ ] **Step 2: `reviewer.md`**: `model: opus`, `tools: Read, Grep, Glob`; чеклист — правило A, fit/агрегаты на valid/test, схема фолдов, формат сабмишна, `code` как не-str, `index=True`; вывод — находки `file:line`, серьёзность.
- [ ] **Step 3: `explorer.md`**: `model: haiku`, `tools: Read, Grep, Glob, Bash`; в описании и теле — запреты `rm`, `mv`, `git commit`, `git push`, запись/правка файлов в репо; parquet — только метаданные, `read_table(columns=…)`, `iter_batches`, не грузить `text` целиком. Если frontmatter поддерживает `disallowedTools` — указать те же запреты.
- [ ] **Step 4: Скиллы:**
  - `validate-submission`: когда — после любого CSV в `outputs/`; `uv run itgorod validate {t1|t2} <csv>`; правила (T1: 505 строк, > 0, порядок id; T2: 10001, `row_id` как в sample, `NNN.NNN.NNN`, только коды train); что делать при типовых ошибках.
  - `time-split-check`: когда — при изменении признаков/валидации T1; правило A; `make check-time-split` и `uv run pytest tests/test_time_split.py -v`; ручной чеклист (лаги через shift по отделу, агрегаты только по train-годам фолда, нет fit на valid).
- [ ] **Step 5: Дополнить `CLAUDE.md`**: когда звать `reviewer`, `explorer`, скиллы.
- [ ] **Step 6: Проверка применения запретов.** Попросить пользователя выполнить `/agents` (слэш-команда мне недоступна) и подтвердить, что `reviewer` и `explorer` видны с нужными моделью и инструментами; затем на отдельном explorer-вызове проверить, что `rm`/`git commit` он не выполняет (на безопасной цели: `rm` несуществующего файла в scratchpad, `git commit` без изменений). Если запреты не применились — записать остаточный риск в `DECISIONS.md` (Task 10, п. 6).
- [ ] **Step 7:** `jq . .claude/settings.json` валиден; хук: `echo '{"tool_input":{"file_path":"src/itgorod/seed.py"}}' | <команда>` → exit 0.
- [ ] **Step 8: Checkpoint** — `log-step`; `chore: reviewer, explorer, validation skills, ruff hook`.

### Task 10: Документация и финальная проверка

**Files:**
- Create: `docs/DECISIONS.md`, `docs/EXPERIMENTS.md`, `notebooks/.gitkeep`
- Modify: `README.md`

- [ ] **Step 1: `DECISIONS.md`** — записи (дата, решение, причина, альтернативы): (1) без Hydra/MLflow/DVC/Docker — текст из спеки §6.6; (2) правило утечки T1 = A, фолды ≤2021→2022, ≤2022→2023; (3) единый CLI; (4) валидация T2 — TBD; (5) группа `dl`, CI `--no-group dl`, CI минимальный и может быть отключён; (6) В `settings.json` только deny на чтение `.env*` и на `git push`; запреты `rm`, `mv`, `git commit` — в описании `explorer` (не глобально, чтобы основной сессии можно было коммитить по команде «коммить»). Ограничение explorer частичное: Bash не закрыть полностью; результат проверки `/agents` — записать; если запреты не применились, остаточный риск принят.
- [ ] **Step 2: `EXPERIMENTS.md`** — таблица `| Дата | Задача | ID | Тип | Конфиг/коммит | Валидация | Метрика | Комментарий |`; строки: T1 `t1-placeholder` тип `placeholder` «—, не оценивается»; T2 `t2-smoke` тип `smoke` «—, не оценивается, 5000 строк train».
- [ ] **Step 3:** `docs/AI_LOG.md` уже ведётся с Task 0b; проверить, что записи есть на каждый чекпойнт.
- [ ] **Step 4: `README.md`** — назначение, `make install`, команды, где лежат сабмишны, ссылки на AGENTS.md и docs.
- [ ] **Step 5: Финальная проверка (спека §7):** `make lint`, `make test`, `make t1`, `make t2-prepare-smoke`, `make t2-smoke`, `make validate`, `make check-time-split`, `uv run pre-commit run --all-files`, имитация CI из Task 8 Step 4 — всё зелёное; `git status` не показывает `data/`, `outputs/`.
- [ ] **Step 6:** Ревью ветки субагентом `reviewer` (или `superpowers:requesting-code-review`); исправить находки.
- [ ] **Step 7: Checkpoint** — `log-step`; `docs: decisions, experiments, readme`; push и проверка CI на GitHub — по отдельной команде пользователя.
