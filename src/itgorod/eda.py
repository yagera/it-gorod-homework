"""Функции EDA для T1. Чистые функции от DataFrame: тестируются на синтетике, графики — в ноутбуке.

Модельные проверки — диагностика гипотез по правилу утечки A (`split_fold`), не бейзлайн.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from itgorod.config import Fold
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


def projects_elasticity(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """α в log y = log r_отдела + α·log(projects): внутриотдельное центрирование (фиксированные
    эффекты), МНК без свободного члена; 95% CI — бутстрэп по отделам. Описательная проверка."""
    d = df[(df[TARGET] > 0) & (df["active_projects_count"] > 0)]
    ly, lx = np.log(d[TARGET]), np.log(d["active_projects_count"])
    g = d["department_id"]
    y = (ly - ly.groupby(g).transform("mean")).to_numpy()
    x = (lx - lx.groupby(g).transform("mean")).to_numpy()
    codes, depts = pd.factorize(g)
    xy, xx = np.bincount(codes, x * y), np.bincount(codes, x * x)  # суммы по отделам
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(depts), len(depts))
        boot.append(xy[pick].sum() / xx[pick].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "alpha": float(xy.sum() / xx.sum()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_rows": len(d),
        "n_depts": len(depts),
    }
