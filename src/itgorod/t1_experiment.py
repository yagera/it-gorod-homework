"""Эксперименты T1: модель A (сглаживание ratio) и A+D (множитель c) против планки
«ratio отдела за прошлый год × проекты» на временных фолдах, на общих строках.

Параметры A выбираются по сумме MAPE двух фолдов — оценка оптимистична.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from itgorod.config import Fold, T1Config, T1Experiment
from itgorod.data import read_t1
from itgorod.eda import projects_elasticity
from itgorod.t1 import (
    BASELINE,
    TARGET,
    RatioParams,
    fit_multiplier,
    predict_ridge,
    predict_target,
)
from itgorod.validation import assert_no_time_leakage, split_fold

COMPARE_KEYS = [
    "mape",
    "n",
    "mape_small",
    "n_small",
    "n_small_worse",
    "share_improved",
    "delta_mape",
    "delta_lo",
    "delta_hi",
]
ModelFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]  # (история, строки) → прогноз таргета
Spec = tuple[RatioParams, bool] | ModelFn
RESULT_COLUMNS = ["model", "valid_year", "w", "phi", "c", *COMPARE_KEYS]


def fold_predictions(train: pd.DataFrame, fold: Fold, specs: dict[str, Spec]) -> pd.DataFrame:
    """Прогнозы моделей на valid-годе фолда. Spec — (RatioParams, применять ли множитель c)
    или функция (история, строки без таргета) → прогноз таргета.
    Только строки с известным таргетом; все прогнозы на них обязаны быть конечными."""
    hist, valid = split_fold(train, fold)
    assert_no_time_leakage(hist, valid)
    valid = valid[valid[TARGET].notna()]
    rows = valid.drop(columns=[TARGET])
    cols = {"department_id": valid["department_id"], "y": valid[TARGET]}
    for name, spec in specs.items():
        if callable(spec):
            c, pred = 1.0, spec(hist, rows)
        else:
            params, use_c = spec
            c = fit_multiplier(hist, params) if use_c else 1.0
            pred = predict_target(hist, rows, params, c)
        cols[f"pred_{name}"] = pred
        cols[f"c_{name}"] = pd.Series(c, index=valid.index)
    out = pd.DataFrame(cols)
    preds = out[[f"pred_{n}" for n in specs]].to_numpy()
    if not np.isfinite(preds).all():
        raise ValueError(f"non-finite predictions on fold {fold}")
    return out.reset_index(drop=True)


def _ape(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.abs(y - p) / y


def compare(
    y: pd.Series,
    pred: pd.Series,
    base: pd.Series,
    small_threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Метрики модели против планки на одних строках; доверительный интервал разницы MAPE —
    95% перцентильный парный бутстрэп по строкам."""
    y, pred, base = (np.asarray(s, dtype=float) for s in (y, pred, base))
    a, b = _ape(y, pred), _ape(y, base)
    small = y < small_threshold
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mape": float(a.mean()),
        "n": int(len(y)),
        "mape_small": float(a[small].mean()) if small.any() else float("nan"),
        "n_small": int(small.sum()),
        "n_small_worse": int((a[small] > b[small] + 1e-12).sum()),
        "share_improved": float((a < b - 1e-12).mean()),
        "delta_mape": float(diff.mean()),
        "delta_lo": float(lo),
        "delta_hi": float(hi),
    }


def select_params(
    train: pd.DataFrame, folds: list[Fold], exp: T1Experiment
) -> tuple[RatioParams, pd.DataFrame]:
    """Сетка w × phi (без множителя), критерий — сумма MAPE по фолдам на общих строках."""
    specs = {
        f"{w}_{phi}": (RatioParams(w, phi, exp.outlier_threshold), False)
        for w in exp.w_grid
        for phi in exp.phi_grid
    }
    per_fold = {}
    for fold in folds:
        fp = fold_predictions(train, fold, specs)
        y = fp["y"].to_numpy()
        per_fold[fold.valid_year] = {
            n: float(_ape(y, fp[f"pred_{n}"].to_numpy()).mean()) for n in specs
        }
    rows = []
    for name, (params, _) in specs.items():
        r = {"w": params.w, "phi": params.phi}
        r.update({f"mape_{yr}": per_fold[yr][name] for yr in per_fold})
        r["mape_sum"] = sum(per_fold[yr][name] for yr in per_fold)
        rows.append(r)
    grid = pd.DataFrame(rows).sort_values(["mape_sum", "w", "phi"], ignore_index=True)
    best = grid.iloc[0]
    return RatioParams(float(best["w"]), float(best["phi"]), exp.outlier_threshold), grid


def run_t1_experiment(cfg: T1Config) -> pd.DataFrame:
    exp = cfg.experiment
    if exp is None:
        raise ValueError("config has no `experiment` section")
    train = read_t1(cfg.paths.train)
    params, grid = select_params(train, cfg.folds, exp)
    specs: dict[str, Spec] = {
        "baseline": (BASELINE, False),
        "A": (params, False),
        "A+D": (params, True),
        "B": lambda h, r: predict_ridge(h, r, params) * r["active_projects_count"],
    }
    rows = []
    for fold in cfg.folds:
        fp = fold_predictions(train, fold, specs)
        for name in specs:
            m = compare(
                fp["y"],
                fp[f"pred_{name}"],
                fp["pred_baseline"],
                exp.small_threshold,
                exp.n_bootstrap,
                cfg.seed,
            )
            c = float(fp[f"c_{name}"].iloc[0])
            p = BASELINE if name == "baseline" else params
            rows.append(
                {"model": name, "valid_year": fold.valid_year, "w": p.w, "phi": p.phi, "c": c, **m}
            )
    res = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    for path, df in ((exp.output, res), (exp.grid_output, grid)):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(res.to_string(index=False))
        print("\nтоп-5 сетки:\n", grid.head().to_string(index=False))
    print(
        f"\nвыбрано: w={params.w}, phi={params.phi} — по сумме MAPE двух фолдов, "
        "оценка оптимистична"
    )
    el = projects_elasticity(train, exp.n_bootstrap, cfg.seed)
    print(
        f"alpha = {el['alpha']:.4f}, 95% CI [{el['ci_lo']:.4f}, {el['ci_hi']:.4f}], "
        f"строк {el['n_rows']}, отделов {el['n_depts']}"
    )
    return res
