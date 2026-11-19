"""Функции EDA для T1. Чистые функции от DataFrame: тестируются на синтетике, графики — в ноутбуке.

Модельные проверки — диагностика гипотез по правилу утечки A (`split_fold`), не бейзлайн.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from itgorod.config import Fold
from itgorod.t1 import BASELINE, predict_ratio
from itgorod.validation import assert_no_time_leakage, split_fold

TARGET = "yearly_cv_processed"
KEYS = ["department_id", "year"]


def numeric_features(df: pd.DataFrame) -> list[str]:
    """Числовые колонки без ключей и таргета."""
    cols = df.select_dtypes("number").columns
    return [c for c in cols if c not in KEYS and c != TARGET]


def quality_report(raw: pd.DataFrame) -> dict:
    """Качество сырого train: пустая строка, строки без ключей, пропуски, дубли, выбросы."""
    empty = raw.isna().all(axis=1)
    no_key = raw[KEYS].isna().any(axis=1)
    clean = raw[~no_key]
    return {
        "n_rows": len(raw),
        "n_fully_empty": int(empty.sum()),
        "n_no_key_not_empty": int((no_key & ~empty).sum()),
        "nan_by_col": clean.isna().sum(),
        "nan_by_col_year": clean.drop(columns=["year"]).isna().groupby(clean["year"]).sum().T,
        "n_dup_dept_year": int(clean.duplicated(KEYS).sum()),
        "n_inbound_zero": int((clean["inbound_applications"] == 0).sum()),
        "inbound_zero_by_year": (clean["inbound_applications"] == 0).groupby(clean["year"]).sum(),
        "n_turnover_gt1": int((clean["turnover_rate"] > 1).sum()),
        "turnover_gt1_depts": sorted(
            clean.loc[clean["turnover_rate"] > 1, "department_id"].unique()
        ),
        "inbound_quantiles": clean["inbound_applications"].quantile([0.5, 0.99]).to_dict(),
        "inbound_top": clean["inbound_applications"].nlargest(4).tolist(),
        "n_rows_any_nan": int(clean.isna().any(axis=1).sum()),
    }


def target_summary(df: pd.DataFrame, small_thresholds=(100, 200)) -> dict:
    """Распределение таргета по годам, доля малых значений, спирмен со связанными признаками."""
    y = df[TARGET].dropna()
    related = ["active_projects_count", "current_staff_count", "total_salary_budget"]
    return {
        "by_year": df.groupby("year")[TARGET].describe(),
        "skew": float(y.skew()),
        "log_skew": float(np.log(y).skew()),
        "small": {t: int((y < t).sum()) for t in small_thresholds},
        "min_values": y.nsmallest(5).tolist(),
        "spearman": df[[TARGET, *related]].corr(method="spearman")[TARGET].drop(TARGET),
        "mape_contrib_small": _small_share_of_mape_weight(y, 200),
    }


def _small_share_of_mape_weight(y: pd.Series, thr: float) -> float:
    """Доля суммарного веса 1/y (вклад в MAPE при равной относительной ошибке), что дают y<thr."""
    w = 1.0 / y
    return float(w[y < thr].sum() / w.sum())


def add_ratio(df: pd.DataFrame) -> pd.DataFrame:
    proj = df["active_projects_count"].where(df["active_projects_count"] > 0)
    return df.assign(ratio=df[TARGET] / proj)


def ratio_stability(df: pd.DataFrame) -> dict:
    """Разброс ratio = таргет / проекты: внутри отдела, по стекам, по годам, тренд внутри отдела."""
    d = add_ratio(df).dropna(subset=["ratio"])
    g = d.groupby("department_id")["ratio"]
    cv_within = g.std() / g.mean()
    slope = d.groupby("department_id").apply(
        lambda x: np.polyfit(x["year"], x["ratio"], 1)[0] if len(x) > 2 else np.nan,
        include_groups=False,
    )
    dev = (d["ratio"] / g.transform("median") - 1).abs()
    return {
        "row_dev_quantiles": dev.quantile([0.5, 0.9, 0.99]).to_dict(),
        "n_rows_dev_gt15pct": int((dev > 0.15).sum()),
        "overall_cv": float(d["ratio"].std() / d["ratio"].mean()),
        "cv_within_dept": cv_within.describe(),
        "between_dept_std": float(g.mean().std()),
        "within_dept_std_mean": float(g.std().mean()),
        "by_year": d.groupby("year")["ratio"].agg(["mean", "median", "std"]),
        "by_stack": d.groupby("tech_stack")["ratio"].agg(["mean", "std", "count"]),
        "slope_per_year": slope.describe(),
        "n_depts_cv_gt10pct": int((cv_within > 0.10).sum()),
    }


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    ok = y_true.notna() & y_pred.notna() & (y_true > 0)
    return float((np.abs(y_true - y_pred)[ok] / y_true[ok]).mean())


def fold_predictions(df: pd.DataFrame, fold: Fold) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Предсказания диагностических моделей для valid-года фолда (индекс — department_id).
    Агрегаты — только по train-годам фолда; из valid берутся проекты и стек того же года."""
    d = add_ratio(df)
    tr, va = split_fold(d, fold)
    assert_no_time_leakage(tr, va)
    va = va.set_index("department_id")
    tr = tr.dropna(subset=["ratio"])
    r_mean = tr.groupby("department_id")["ratio"].mean()
    r_last = tr.sort_values("year").groupby("department_id")["ratio"].last()
    r_stack = tr.groupby("tech_stack")["ratio"].mean()
    lag = tr[tr["year"] == fold.valid_year - 1].set_index("department_id")[TARGET]
    proj, ids = va["active_projects_count"], va.index
    return va[TARGET], {
        "lag_target": lag.reindex(ids),
        "ratio_dept_mean": r_mean.reindex(ids) * proj,
        "ratio_dept_last": r_last.reindex(ids) * proj,
        "ratio_stack_mean": va["tech_stack"].map(r_stack) * proj,
    }


def ratio_model_diagnostics(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """MAPE по временным фолдам на ОБЩИХ строках (где есть таргет и все предсказания);
    строка `n_rows` — размер общего набора."""
    out = {}
    for f in folds:
        y, preds = fold_predictions(df, f)
        common = y.notna() & pd.concat(preds, axis=1).notna().all(axis=1)
        col = {name: mape(y[common], p[common]) for name, p in preds.items()}
        col["n_rows"] = int(common.sum())
        out[f.valid_year] = col
    return pd.DataFrame(out)


def identities(df: pd.DataFrame) -> pd.DataFrame:
    """Точные тождества между признаками одного года: для них можно восстанавливать пропуски."""
    checks = {
        "budget = k*staff*salary": df["total_salary_budget"]
        / (df["current_staff_count"] * df["avg_dev_salary"]),
        "turnover = (hired+fired)/staff": df["turnover_rate"]
        * df["current_staff_count"]
        / (df["devs_hired"] + df["devs_fired"]),
    }
    return pd.DataFrame({k: v.describe() for k, v in checks.items()}).T


def feature_target_signal(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Выражается ли таргет формулой из одного признака того же года: Spearman и MAPE модели
    log1p(y) ~ log1p(x), обученной на train фолда и оценённой на valid (среднее по фолдам)."""
    d = df.dropna(subset=[TARGET])
    rows = []
    for c in numeric_features(d):
        rho = d[[c, TARGET]].corr(method="spearman").iloc[0, 1]
        errs = []
        for f in folds:
            tr, va = split_fold(d.dropna(subset=[c]), f)
            assert_no_time_leakage(tr, va)
            b, a = np.polyfit(np.log1p(tr[c].clip(lower=0)), np.log(tr[TARGET]), 1)
            pred = np.exp(a + b * np.log1p(va[c].clip(lower=0)))
            errs.append(mape(va[TARGET], pred))
        rows.append({"feature": c, "spearman": rho, "mape_loglin": float(np.mean(errs))})
    return pd.DataFrame(rows).set_index("feature").sort_values("spearman", key=abs, ascending=False)


def drift_table(train: pd.DataFrame, test: pd.DataFrame, year: int) -> pd.DataFrame:
    """Сдвиг признаков: train[year] против test — средние, стандартизованный сдвиг, KS."""
    base = train[train["year"] == year]
    rows = []
    for c in numeric_features(test):
        a, b = base[c].dropna(), test[c].dropna()
        ks = ks_2samp(a, b, method="asymp")
        rows.append(
            {
                "feature": c,
                "mean_train": a.mean(),
                "mean_test": b.mean(),
                "shift_pct": 100 * (b.mean() / a.mean() - 1),
                "std_shift": (b.mean() - a.mean()) / a.std(),
                "ks": ks.statistic,
                "p": ks.pvalue,
            }
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("ks", ascending=False)


def stack_constancy(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """tech_stack постоянна ли по отделу (train и test), совпадает ли test с train."""
    n_unique = train.groupby("department_id")["tech_stack"].nunique()
    last = train.sort_values("year").groupby("department_id")["tech_stack"].last()
    match = (
        test.set_index("department_id")["tech_stack"]
        == last.reindex(test["department_id"]).to_numpy()
    )
    return {
        "n_depts_multi_stack": int((n_unique > 1).sum()),
        "n_stacks": int(train["tech_stack"].nunique()),
        "test_matches_train": float(match.mean()),
        "depts_per_stack": train.drop_duplicates("department_id")["tech_stack"].value_counts(),
    }


def _fe_sums(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Суммы по отделам для МНК log y ~ log projects с фиксированными эффектами отдела:
    (Σxy, Σxx по отделам, число строк, число отделов)."""
    d = df[(df[TARGET] > 0) & (df["active_projects_count"] > 0)]
    ly, lx = np.log(d[TARGET]), np.log(d["active_projects_count"])
    g = d["department_id"]
    y = (ly - ly.groupby(g).transform("mean")).to_numpy()
    x = (lx - lx.groupby(g).transform("mean")).to_numpy()
    codes, depts = pd.factorize(g)
    return np.bincount(codes, x * y), np.bincount(codes, x * x), len(d), len(depts)


def _slope(xy: np.ndarray, xx: np.ndarray) -> float:
    return float(xy.sum() / xx.sum()) if xx.sum() > 0 else float("nan")


def elasticity_alpha(df: pd.DataFrame) -> float:
    """Точечная оценка α (NaN, если проекты внутри отделов не меняются)."""
    xy, xx, _, _ = _fe_sums(df)
    return _slope(xy, xx)


def projects_elasticity(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """α в log y = log r_отдела + α·log(projects): внутриотдельное центрирование (фиксированные
    эффекты), МНК без свободного члена; 95% CI — бутстрэп по отделам. Описательная проверка."""
    xy, xx, n_rows, n_depts = _fe_sums(df)
    if n_depts == 0:
        return {"alpha": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_rows": 0, "n_depts": 0}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_depts, n_depts)
        boot.append(_slope(xy[pick], xx[pick]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if not np.isnan(boot).all() else (np.nan,) * 2
    return {
        "alpha": _slope(xy, xx),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_rows": n_rows,
        "n_depts": n_depts,
    }


# ---------------------------------------------------------------------------
# Разбор малых значений таргета: где и почему ошибается планка «ratio за последний год × проекты».


def last_year_rows(df: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Строки valid-года фолда с известным таргетом и прогнозом планки. Колонки `p_*` (последний
    год train с ratio: год, проекты, таргет, ratio) и `pred` считаются только по train-годам;
    таргет valid-года входит лишь в `y`, `ratio`, `ape`, `dly`, `dlr` (оценка, не прогноз)."""
    d = add_ratio(df)
    tr, va = split_fold(d, fold)
    assert_no_time_leakage(tr, va)
    last = tr.dropna(subset=["ratio"]).sort_values("year").groupby("department_id").tail(1)
    last = last.set_index("department_id")
    v = va[va[TARGET].notna()].set_index("department_id")
    out = pd.DataFrame(
        {
            "y": v[TARGET],
            "proj": v["active_projects_count"],
            "staff": v["current_staff_count"],
            "ratio": v["ratio"],
            "p_year": last["year"].reindex(v.index),
            "p_proj": last["active_projects_count"].reindex(v.index),
            "p_y": last[TARGET].reindex(v.index),
            "p_ratio": last["ratio"].reindex(v.index),
        }
    )
    out["pred"] = out["p_ratio"] * out["proj"]
    out["ape"] = (out["pred"] - out["y"]).abs() / out["y"]
    out["dlp"] = np.log(out["proj"] / out["p_proj"])
    out["dly"] = np.log(out["y"] / out["p_y"])
    out["dlr"] = np.log(out["ratio"] / out["p_ratio"])
    return out.dropna(subset=["pred"]).reset_index()


def small_rows(df: pd.DataFrame, fold: Fold, threshold: float = 200) -> pd.DataFrame:
    rows = last_year_rows(df, fold)
    return rows[rows["y"] < threshold].reset_index(drop=True)


def _group_labels(values: pd.Series, bins: list[float]) -> pd.Series:
    labels = [f"({lo:g}, {hi:g}]" for lo, hi in zip(bins[:-1], bins[1:], strict=True)]
    return pd.cut(values, bins, labels=labels, include_lowest=True)


def _dept_group(df: pd.DataFrame, bins: list[float]) -> pd.Series:
    """Группа размера отдела по среднему числу проектов в строках с ratio (общее правило для
    `elasticity_by_size` и `predict_elastic`, чтобы печатаемые α были теми, что применяются)."""
    h = add_ratio(df).dropna(subset=["ratio"])
    return _group_labels(h.groupby("department_id")["active_projects_count"].mean(), bins)


def elasticity_by_size(
    df: pd.DataFrame, max_year: int, bins: list[float], n_boot: int, seed: int
) -> pd.DataFrame:
    """H1 (intercept ⇒ α < 1 у малых): α по группам отделов по среднему числу проектов,
    по годам <= max_year (train-годы фолда)."""
    t = df[df["year"] <= max_year]
    group = _dept_group(t, bins)
    rows = []
    for label in group.cat.categories:
        ids = group.index[group == label]
        if len(ids) == 0:
            continue
        r = projects_elasticity(t[t["department_id"].isin(ids)], n_boot, seed)
        rows.append(
            {
                "group": label,
                **{k: r[k] for k in ("n_depts", "n_rows", "alpha")},
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
            }
        )
    return pd.DataFrame(rows)


def dept_intercepts(df: pd.DataFrame, min_points: int = 4) -> pd.DataFrame:
    """H1 напрямую: по отделу МНК y = a + b·projects. `a_rel` = a / среднее y, `t` = a / se(a).
    Описательно, по всем годам (не прогноз); по 4 точкам мощность низкая."""
    rows = []
    for did, g in df.dropna(subset=[TARGET, "active_projects_count"]).groupby("department_id"):
        x, y = g["active_projects_count"].to_numpy(), g[TARGET].to_numpy()
        if len(g) < min_points or len(np.unique(x)) < 3:
            continue
        b, a = np.polyfit(x, y, 1)
        sxx = ((x - x.mean()) ** 2).sum()
        rss = ((y - a - b * x) ** 2).sum()
        se_a = np.sqrt(rss / (len(x) - 2) * (1 / len(x) + x.mean() ** 2 / sxx))
        rows.append(
            {
                "department_id": did,
                "mean_proj": x.mean(),
                "a_rel": a / y.mean(),
                "t": a / se_a if se_a > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def predict_elastic(
    history: pd.DataFrame, rows: pd.DataFrame, bins: list[float] | None = None
) -> pd.Series:
    """Кандидат-поправка (H1): y_last · (proj / p_proj)^α, α — FE-эластичность по history.
    `bins=None` — одна α на всех; иначе α по группам отделов (среднее число проектов в history).
    При α = 1 совпадает с планкой. Отделы без истории — как у планки (fallback-цепочка)."""
    ratio = predict_ratio(history, rows, BASELINE)  # заодно проверяет правило A
    h = add_ratio(history).dropna(subset=["ratio"]).sort_values("year")
    by_dept = h.groupby("department_id")["active_projects_count"]
    p_proj, ids = by_dept.last(), rows["department_id"]
    if bins is None:
        alpha = pd.Series(elasticity_alpha(history), index=ids.index)
    else:
        group = _dept_group(history, bins)
        alpha_by_group = {
            g: elasticity_alpha(history[history["department_id"].isin(group.index[group == g])])
            for g in group.cat.categories
        }
        alpha = ids.map(group).map(alpha_by_group).astype(float)
    alpha = alpha.fillna(1.0)
    proj = rows["active_projects_count"]
    factor = (proj / ids.map(p_proj)) ** (alpha - 1.0)
    factor = factor.where(np.isfinite(factor), 1.0)  # проекты = 0 или нет истории
    return (ratio * proj * factor).rename(TARGET)
