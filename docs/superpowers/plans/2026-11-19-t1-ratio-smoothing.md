# T1: сглаживание ratio (A) + множитель MAPE (D) — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (выбрано пользователем: TDD в этой сессии). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** обойти планку «ratio отдела за прошлый год × active_projects_count» (MAPE 2,8% / 3,0% на фолдах 2022 / 2023) сглаживанием ratio по отделу (A) и множителем c (D), с честным сравнением по строкам.

**Architecture:** модель — чистые функции в `src/itgorod/t1.py` (`predict_ratio`, `predict_target`, `fit_multiplier`), принимают только историю (годы < целевого) и строки целевого года без таргета. Раннер `src/itgorod/t1_experiment.py` гоняет оба фолда на общих строках, считает метрики и пишет `outputs/experiments.csv`. Проверка α — функция в `src/itgorod/eda.py`.

**Tech Stack:** pandas, numpy, pytest; без новых зависимостей.

**Spec:** запрос пользователя от 2026-11-19 (выбор A + D, шаги 1–5, условие для B), контекст — `docs/DECISIONS.md`, запись 7. Отдельного design-doc нет: пользователь перешёл к плану сам.

## Global Constraints

- Правило утечки A: для года Y — признаки года Y и любые данные лет < Y; нельзя таргет года Y и данные лет > Y. Все агрегаты — только по train-годам фолда.
- Параметров модели A — не больше двух: `w` (смесь) и `phi` (затухание тренда). Порог выбросов 0,15 — константа из конфига, не подбирается.
- Исключение из истории: годы, где |ratio / медиана ratio отдела − 1| > 0,15.
- Fallback: прогноз модели → ratio отдела за последний год → медиана ratio по `tech_stack` → общая медиана.
- Множитель c подбирается только на train-годах фолда. Веса 1/y не используются.
- Выбор параметров — по сумме MAPE двух фолдов; в `EXPERIMENTS.md` и выводе писать, что оценка оптимистична.
- Срез малых: y < 200.
- Коммитов без команды пользователя нет. Не трогать `run_t1` и `outputs/task-1.csv` (сабмишн — отдельный шаг).
- Seed — `cfg.seed` (42).

## Модель A (решения, которые исполнитель не выводит сам)

Для отдела по чистым (не выброшенным) годам истории:
- `r_last` — ratio последнего чистого года; `r_mean` — среднее чистых; `slope` — МНК-наклон ratio по году при ≥ 2 чистых точках, иначе 0.
- `r_hat = w · (r_last + phi · slope) + (1 − w) · r_mean`.
- Если чистых точек нет или `r_hat <= 0` → fallback-цепочка с её первого шага (сырой ratio отдела за последний год с ratio).
- Медианы для fallback — по всем строкам истории с ratio (без исключения выбросов).
- Планка = `RatioParams(w=1.0, phi=0.0, outlier_threshold=inf)`: сырой ratio последнего года × проекты.

Множитель D: внутренний бэктест по истории. Для каждого года `t` истории, у которого есть хотя бы один более ранний год, прогноз `predict_target(history[year < t], history[year == t] без таргета)`. При MAPE `|y − c·p| / y = (p/y) · |y/p − c|`, поэтому оптимальный `c` — **взвешенная медиана y/p с весами p/y**. Нет внутренних лет → c = 1.0.

## Review Focus

1. Отдел, у которого все годы истории — выбросы (на фолде 2022 всего 2 точки, обе могут выйти за 15%) → должен сработать сырой ratio последнего года, а не NaN. Тест в Task 1.
2. Тренд уводит ratio в ≤ 0 (наклон до −3,6 при ratio ≈ 9) → fallback, прогноз > 0. Тест в Task 1.
3. Отдел или стек, которых нет в истории → медиана стека / общая медиана. Тест в Task 1.
4. Строки целевого года с непоследовательным индексом и перемешанным порядком → результат выровнен по индексу `rows`. Тест в Task 1.
5. `active_projects_count` NaN в строке целевого года → прогноз NaN, раннер такие строки исключает (на реальных данных они совпадают с NaN таргета) и проверяет, что на общих строках прогнозы конечны. Тест в Task 1 и Task 4.

---

### Task 1: Модель A — `predict_ratio`, `predict_target`

**Files:**
- Modify: `src/itgorod/t1.py` (добавить; `predict_placeholder` и `run_t1` не трогать)
- Test: `tests/test_t1_ratio.py` (новый; своя фикстура с `active_projects_count`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class RatioParams: w: float; phi: float; outlier_threshold: float = 0.15`
  - `BASELINE = RatioParams(w=1.0, phi=0.0, outlier_threshold=float("inf"))`
  - `predict_ratio(history: pd.DataFrame, rows: pd.DataFrame, params: RatioParams) -> pd.Series` — индекс = `rows.index`, имя `"ratio"`.
  - `predict_target(history, rows, params, c: float = 1.0) -> pd.Series` = `c · predict_ratio · rows.active_projects_count`.
  - Охрана: `LeakageError`, если `history.year.max() >= rows.year.min()`; `ValueError`, если в `rows` есть колонка `yearly_cv_processed`.

- [ ] **Step 1: Фикстура и падающие тесты** в `tests/test_t1_ratio.py`. Фикстура `ratio_frame`: 6 отделов × 2020–2023, стеки `A`/`B`, `active_projects_count = 10 + 5d + (year − 2020)`, таргет = `ratio_d(year) × проекты`, где ratio отдела задан явно в тесте. Тесты:
  - `test_baseline_equals_last_year_ratio` — `BASELINE` даёт ratio 2022 года отдела × проекты 2023.
  - `test_blend_and_trend` — ratio отдела 20, 21, 22 (2020–2022): при `w=1, phi=1` → 23; при `w=0` → 21; при `w=0.5, phi=0` → 21.5.
  - `test_outlier_year_excluded` — ratio 20, 20, 30 (30 — выброс > 15% от медианы 20): при `w=1, phi=0` → 20.
  - `test_all_outliers_fall_back_to_raw_last` — две точки 10 и 20 (обе на 33% от медианы 15) → 20.
  - `test_nonpositive_trend_falls_back` — ratio 9, 3 (`w=1, phi=1`, порог `inf`) → r_hat = −3 ≤ 0 → 3.
  - `test_unknown_dept_uses_stack_then_overall_median` — новый отдел со стеком `A` → медиана ratio стека `A`; новый стек `C` → общая медиана.
  - `test_rows_alignment` — `rows` перемешаны и с индексом `[100, 7, 42, ...]`; результат `.index.equals(rows.index)`.
  - `test_nan_projects_gives_nan_target` — NaN в проектах → NaN в `predict_target`, остальные конечны.
  - `test_guards` — история с годом ≥ года `rows` → `LeakageError`; `rows` с таргетом → `ValueError`.

- [ ] **Step 2: Run** `uv run pytest tests/test_t1_ratio.py -q` → FAIL (`ImportError: RatioParams`).

- [ ] **Step 3: Реализовать** `RatioParams`, `BASELINE`, `predict_ratio`, `predict_target` в `src/itgorod/t1.py` по разделу «Модель A». Ratio строки истории = таргет / проекты при проектах > 0, иначе NaN.

- [ ] **Step 4: Run** `uv run pytest tests/test_t1_ratio.py tests/test_t1.py -q` → PASS.

- [ ] **Step 5:** без коммита (ждём команды пользователя).

### Task 2: Множитель D — `fit_multiplier`

**Files:**
- Modify: `src/itgorod/t1.py`
- Test: `tests/test_t1_ratio.py`

**Interfaces:**
- Consumes: `predict_target`, `RatioParams` (Task 1).
- Produces: `weighted_median(values: np.ndarray, weights: np.ndarray) -> float`; `fit_multiplier(history: pd.DataFrame, params: RatioParams) -> float`.

- [ ] **Step 1: Падающие тесты:**
  - `test_weighted_median` — `weighted_median([1, 2, 3], [1, 1, 5]) == 3`; `weighted_median([1, 2], [1, 1])` ∈ {1, 2} (нижняя медиана: берём первое значение, где накопленный вес ≥ половины).
  - `test_multiplier_recovers_scale` — история, где каждый год ratio отдела ровно в 1,05 раза выше прошлого, `BASELINE` → прогноз занижен в 1,05 раза → `fit_multiplier ≈ 1.05` (abs 1e-9).
  - `test_multiplier_one_year_history` — история из одного года → `1.0`.
  - `test_multiplier_uses_only_history` — `fit_multiplier(history)` не меняется, если в исходной таблице умножить таргет лет ≥ valid_year на 5 (история берётся через `split_fold`).

- [ ] **Step 2: Run** `uv run pytest tests/test_t1_ratio.py -q -k "multiplier or weighted"` → FAIL.

- [ ] **Step 3: Реализовать** `weighted_median` и `fit_multiplier` по разделу «Модель A» (внутренний бэктест, строки с NaN y или p пропускаются).

- [ ] **Step 4: Run** `uv run pytest tests/test_t1_ratio.py -q` → PASS.

- [ ] **Step 5:** без коммита.

### Task 3: Проверка α — `projects_elasticity`

**Files:**
- Modify: `src/itgorod/eda.py`
- Test: `tests/test_eda.py`

**Interfaces:**
- Produces: `projects_elasticity(df: pd.DataFrame, n_boot: int, seed: int) -> dict` с ключами `alpha`, `ci_lo`, `ci_hi`, `n_rows`, `n_depts`.
- Метод: строки с y > 0 и проектами > 0; внутриотдельное центрирование `log y` и `log projects` (фиксированные эффекты отдела = log r); МНК-наклон без свободного члена. CI 95% — перцентили бутстрэпа **по отделам** (кластерный), `np.random.default_rng(seed)`.
- Правило «α заметно отличается от 1»: CI не содержит 1 **и** |α − 1| ≥ 0,05. Описательная проверка по всем train-годам (2020–2023), в прогноз не идёт.

- [ ] **Step 1: Падающие тесты:**
  - `test_elasticity_recovers_alpha` — 30 отделов × 4 года, `y = r_d · proj^1.2 · exp(шум 0,01)` → `|alpha − 1.2| < 0.02`, `ci_lo < 1.2 < ci_hi`.
  - `test_elasticity_one_for_pure_ratio` — `_frame()` из `tests/test_eda.py` (ratio постоянен) → `alpha ≈ 1` (abs 1e-9).
  - `test_elasticity_deterministic` — два вызова с одним seed дают одинаковый словарь.

- [ ] **Step 2: Run** `uv run pytest tests/test_eda.py -q -k elasticity` → FAIL.
- [ ] **Step 3: Реализовать** `projects_elasticity`.
- [ ] **Step 4: Run** `uv run pytest tests/test_eda.py -q` → PASS.
- [ ] **Step 5:** без коммита.

### Task 4: Раннер экспериментов + конфиг + CLI

**Files:**
- Create: `src/itgorod/t1_experiment.py`
- Modify: `configs/t1.yaml`, `src/itgorod/config.py`, `src/itgorod/cli.py`, `Makefile`
- Test: `tests/test_t1_experiment.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `RatioParams`, `BASELINE`, `predict_target`, `fit_multiplier` (Tasks 1–2); `projects_elasticity` (Task 3); `split_fold`, `assert_no_time_leakage`.
- Produces:
  - Конфиг, секция `experiment` → `T1Config.experiment: T1Experiment(outlier_threshold: float, w_grid: list[float], phi_grid: list[float], small_threshold: float, n_bootstrap: int, output: Path, grid_output: Path)`. Значения: `0.15`, `w_grid` 0.0…1.0 шаг 0.1, `phi_grid` `[0, 0.25, 0.5, 0.75, 1]`, `200`, `2000`, `outputs/experiments.csv`, `outputs/t1_grid.csv`.
  - `fold_predictions(train, fold, specs: dict[str, tuple[RatioParams, bool]]) -> pd.DataFrame` — колонки `department_id`, `y`, `pred_<name>`; bool = применять ли `fit_multiplier`; только строки valid с `y.notna()` и всеми конечными прогнозами (иначе `ValueError`); плюс колонка `c_<name>`.
  - `compare(y, pred, base, small_threshold, n_boot, seed) -> dict`: `mape`, `n`, `mape_small`, `n_small`, `n_small_worse` (APE модели > APE планки на малых строках), `share_improved` (доля строк, где APE модели < APE планки), `delta_mape` (модель − планка), `delta_lo`, `delta_hi` (95% перцентильный парный бутстрэп по строкам).
  - `select_params(train, folds, exp) -> tuple[RatioParams, pd.DataFrame]` — сетка `w × phi` без D, критерий — сумма MAPE по фолдам на общих строках; таблица сетки → `grid_output`.
  - `run_t1_experiment(cfg) -> pd.DataFrame` — модели `baseline` (`BASELINE`, без c), `A` (выбранные параметры, без c), `A+D` (с c); строка на (модель, фолд) с колонками `model, valid_year, w, phi, c` + поля `compare`; пишет `experiment.output`; печатает таблицу, выбранные параметры, пометку «параметры выбраны по сумме двух фолдов — оценка оптимистична» и результат `projects_elasticity`.
  - CLI: `itgorod t1 experiment --config configs/t1.yaml`; Makefile: `t1-experiment`.

- [ ] **Step 1: Падающие тесты** (`tests/test_t1_experiment.py`, синтетика с шумом ratio и трендом):
  - `test_config_has_experiment_section` (в `tests/test_config.py`) — значения из конфига как выше; неизвестный ключ в `experiment` → `ValueError`.
  - `test_compare_counts` — ручной пример из 4 строк (2 малые): проверить `n_small`, `n_small_worse`, `share_improved`, `delta_mape`; при `pred == base` → `delta_mape == 0`, `delta_lo == delta_hi == 0`.
  - `test_compare_bootstrap_deterministic` — одинаковый seed → одинаковые `delta_lo/hi`.
  - `test_fold_predictions_common_rows` — строка valid с NaN y исключена; у всех моделей одинаковые строки.
  - `@pytest.mark.parametrize("fold", FOLDS)` `test_fold_predictions_ignore_valid_and_future` — умножить таргет лет ≥ `fold.valid_year` на 5 и признаки лет > `valid_year` на 3 → `pred_*` и `c_*` не меняются.
  - `test_select_params_returns_grid_point` — выбранные `w, phi` из сетки; длина таблицы сетки = `len(w_grid) · len(phi_grid)`.
  - `test_run_writes_csv` — `tmp_path`, CSV читается, колонки ровно как в Interfaces, 3 модели × 2 фолда = 6 строк.

- [ ] **Step 2: Run** `uv run pytest tests/test_t1_experiment.py tests/test_config.py -q` → FAIL.
- [ ] **Step 3: Реализовать** конфиг, `t1_experiment.py`, CLI-подкоманду и цель Makefile.
- [ ] **Step 4: Run** `make lint && make test` → всё зелёное.
- [ ] **Step 5:** без коммита.

### Task 5: Прогон, документы, решение по B

**Files:**
- Modify: `docs/EXPERIMENTS.md`, `docs/DECISIONS.md`, `docs/AI_LOG.md`

- [ ] **Step 1: Run** `make t1-experiment` → печатает таблицу; `outputs/experiments.csv` и `outputs/t1_grid.csv` созданы.
- [ ] **Step 2: Скилл `time-split-check`** (изменились признаки и модель T1): `make check-time-split` → OK, плюс ручной чеклист скилла по `t1.py` и `t1_experiment.py`.
- [ ] **Step 3: `EXPERIMENTS.md`** — строки: `t1-ratio-last` (тип `baseline`), `t1-A`, `t1-A+D` (тип `experiment`); в комментарии — параметры, c по фолдам, «оценка оптимистична: параметры выбраны по сумме двух фолдов».
- [ ] **Step 4: `DECISIONS.md`** — запись 8: C (бустинг на остатке) рассмотрен и отклонён: мало строк (~500 / ~1000 на фолд), шум ratio ~2% (нижний предел), риск подгонки под valid. Запись 9: результат A + D, α с интервалом, решение по B.
- [ ] **Step 5: Решение по B.** B (Ridge на log ratio) реализуется, только если A + D **не** ниже планки на обоих фолдах **или** α заметно отличается от 1 (правило из Task 3). Иначе — остановиться и показать результаты. В обоих случаях следующий шаг — за пользователем.
- [ ] **Step 6: Субагент `reviewer`** — ревью на утечки (`t1.py`, `t1_experiment.py`, тесты); правки по замечаниям.
- [ ] **Step 7: Скилл `log-step`**, показать итоговую таблицу, остановиться. Без коммита.

### Task 6 (добавлена после прогона Task 5, условие для B выполнено): Ridge на log(ratio)

Условие сработало: A+D не ниже планки на обоих фолдах (Δ A+D: 2022 +0,01 п.п., 2023 +0,11 п.п.; у A: −0,01 / +0,03). α = 1,04, CI [0,92; 1,16] — от 1 не отличается.

**Files:** Modify `src/itgorod/t1.py`, `src/itgorod/t1_experiment.py`; Test `tests/test_t1_ratio.py`, `tests/test_t1_experiment.py`.

**Модель B** (`predict_ridge(history, rows, params_a: RatioParams, ridge_alpha: float = 1.0) -> pd.Series`, ratio строк, индекс = `rows.index`):
- Таргет на обучении: `z_t = log r_t − log r_{t−1}` (изменение ratio к прошлому году). Прогноз: `r_{Y−1} · exp(ẑ)`.
- Обучающие строки — только годы истории `t`, у которых есть `r_{t−1}`; признаки года `t` — из той же истории (без строк целевого года).
- Признаки (для обучающего года t и для целевого Y одинаково; **признак `a = log r̂A(t) − log r_{t−1}` из первой версии удалён** — на обучении почти константа, переполнение exp; ẑ клипуется диапазоном z обучения): `dlog_proj`, `dlog_staff`, `dlog_inbound` (0 → NaN), `inbound_missing` (флаг), `tech_brand_rating`, `dbrand`. NaN в признаках → 0 (после дельт; флаг для inbound).
- `StandardScaler` + `Ridge(alpha=1.0)` обучаются только на обучающих строках. ridge_alpha не подбирается (нет valid для подбора).
- Отдел без `r_{Y−1}` → `predict_ratio(history, rows, params_a)`. Меньше 20 обучающих строк → тоже `predict_ratio`.
- Охрана — как у `predict_ratio`.
- Раннер: spec расширяется до `tuple[RatioParams, bool] | Callable[[hist, rows], pd.Series]` (callable возвращает прогноз таргета); модель `B` = Ridge с выбранными параметрами A, без c.

**Тесты:** `test_ridge_learns_signal` (z = 0,5·dlog_inbound + шум 0,001 → MAPE ratio < 0,5%); `test_ridge_falls_back_without_last_ratio`; `test_ridge_guards_and_alignment`; в `test_fold_predictions_ignore_valid_and_future` — добавить B в SPECS; `test_run_writes_csv` → 4 модели × 2 фолда = 8 строк.
