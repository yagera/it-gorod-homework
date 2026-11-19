"""T1: прогноз `yearly_cv_processed` = ratio отдела × `active_projects_count`.

Сабмишн (`run_t1`) — планка `BASELINE`: ratio отдела за последний год истории × проекты года
прогноза (выбор — docs/DECISIONS.md, запись 10). Модели A (сглаживание ratio), D (множитель c)
и B (Ridge) — для экспериментов (`itgorod.t1_experiment`), в сабмишн не идут.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from itgorod.config import T1Config
from itgorod.data import read_t1
from itgorod.seed import set_seed
from itgorod.validation import LeakageError, assert_features_for_year

TARGET = "yearly_cv_processed"


# ---------------------------------------------------------------------------
# Модель A: сглаживание ratio = yearly_cv_processed / active_projects_count по отделу.
# Модель D: множитель c, подобранный внутренним бэктестом только по истории.
# Функции получают историю (годы < целевого) и строки целевого года без таргета.


@dataclass(frozen=True)
class RatioParams:
    """r_hat = w·(r_last + phi·slope) + (1 − w)·r_mean по «чистым» годам отдела."""

    w: float
    phi: float
    outlier_threshold: float = 0.15


BASELINE = RatioParams(w=1.0, phi=0.0, outlier_threshold=float("inf"))


def _history_ratio(history: pd.DataFrame) -> pd.DataFrame:
    proj = history["active_projects_count"].where(history["active_projects_count"] > 0)
    h = history.assign(ratio=history[TARGET] / proj)
    return h.dropna(subset=["ratio"])[["department_id", "year", "tech_stack", "ratio"]]


def _dept_model_ratio(h: pd.DataFrame, params: RatioParams) -> pd.Series:
    """Прогноз ratio по отделам из чистой истории; NaN, если чистых лет нет или r_hat <= 0."""
    dev = (h["ratio"] / h.groupby("department_id")["ratio"].transform("median") - 1).abs()
    clean = h[dev <= params.outlier_threshold].sort_values("year")
    g = clean.groupby("department_id")
    r_last, r_mean = g["ratio"].last(), g["ratio"].mean()

    def _slope(x: pd.DataFrame) -> float:
        return float(np.polyfit(x["year"], x["ratio"], 1)[0]) if len(x) >= 2 else 0.0

    slope = g[["year", "ratio"]].apply(_slope) if len(clean) else pd.Series(dtype=float)
    r_hat = params.w * (r_last + params.phi * slope) + (1 - params.w) * r_mean
    return r_hat.where(r_hat > 0)


def predict_ratio(history: pd.DataFrame, rows: pd.DataFrame, params: RatioParams) -> pd.Series:
    """Ratio для строк целевого года. Fallback: модель → сырой ratio отдела за последний год →
    медиана ratio по tech_stack → общая медиана (медианы — по всей истории)."""
    if TARGET in rows.columns:
        raise ValueError(f"rows must not contain {TARGET}")
    if history.empty:
        raise ValueError("history is empty: nothing to forecast from")
    if history["year"].max() >= rows["year"].min():
        raise LeakageError(
            [f"history year {history['year'].max()} >= target year {rows['year'].min()}"]
        )
    h = _history_ratio(history)
    raw_last = h.sort_values("year").groupby("department_id")["ratio"].last()
    stack_median = h.groupby("tech_stack")["ratio"].median()
    ids = rows["department_id"]
    out = ids.map(_dept_model_ratio(h, params))
    out = out.fillna(ids.map(raw_last))
    out = out.fillna(rows["tech_stack"].map(stack_median))
    return out.fillna(h["ratio"].median()).astype(float).rename("ratio")


def predict_target(
    history: pd.DataFrame, rows: pd.DataFrame, params: RatioParams, c: float = 1.0
) -> pd.Series:
    return (c * predict_ratio(history, rows, params) * rows["active_projects_count"]).rename(TARGET)


def predict_baseline(train: pd.DataFrame, test: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Планка T1: ratio отдела за последний год истории × `active_projects_count` из `test`.
    Fallback: ratio более раннего года → медиана ratio по tech_stack → общая медиана.
    Порядок строк как в `test`; нефинитный или неположительный прогноз → ValueError."""
    assert_features_for_year(pd.concat([train, test], ignore_index=True), target_year, TARGET)
    history = train[train["year"] < target_year]
    values = predict_target(history, test, BASELINE)
    bad = ~np.isfinite(values.to_numpy()) | (values.to_numpy() <= 0)
    if bad.any():
        ids = test.loc[bad, "department_id"].tolist()[:5]
        raise ValueError(f"{int(bad.sum())} non-finite or non-positive prediction(s), e.g. {ids}")
    return pd.DataFrame(
        {"department_id": test["department_id"].to_numpy(), TARGET: values.to_numpy()}
    )


def run_t1(cfg: T1Config) -> Path:
    set_seed(cfg.seed)
    train = read_t1(cfg.paths.train)
    test = read_t1(cfg.paths.test)
    sample = pd.read_csv(cfg.paths.sample, usecols=["department_id"])
    pred = predict_baseline(train, test, cfg.target_year)
    out = sample.merge(pred, on="department_id", how="left")
    if out[TARGET].isna().any():
        missing = out.loc[out[TARGET].isna(), "department_id"].tolist()[:5]
        raise ValueError(f"no prediction for sample department_id(s), e.g. {missing}")
    cfg.paths.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.paths.output, index=False)
    return cfg.paths.output


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Нижняя взвешенная медиана: первое значение, где накопленный вес >= половины."""
    order = np.argsort(values, kind="stable")
    v, cw = np.asarray(values)[order], np.cumsum(np.asarray(weights)[order])
    return float(v[np.searchsorted(cw, cw[-1] / 2)])


def fit_multiplier(history: pd.DataFrame, params: RatioParams) -> float:
    """c = argmin MAPE(y, c·p) по внутреннему бэктесту истории = взвешенная медиана y/p
    с весами p/y. Нет лет с предысторией → 1.0."""
    ys, ps = [], []
    for t in sorted(history["year"].unique())[1:]:
        cur = history[history["year"] == t]
        p = predict_target(history[history["year"] < t], cur.drop(columns=[TARGET]), params)
        ys.append(cur[TARGET].to_numpy())
        ps.append(p.to_numpy())
    if not ys:
        return 1.0
    y, p = np.concatenate(ys), np.concatenate(ps)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    if not ok.any():
        return 1.0
    return weighted_median(y[ok] / p[ok], p[ok] / y[ok])


# ---------------------------------------------------------------------------
# Модель B: Ridge на изменении log ratio к прошлому году; признаки года t и t−1.

# Признака «A-прогноз против прошлого года» нет: его разброс зависит от длины истории
# (на годах с одним предыдущим годом он тождественно 0), и Ridge экстраполирует.
RIDGE_FEATURES = [
    "dlog_proj",
    "dlog_staff",
    "dlog_inbound",
    "inbound_missing",
    "brand",
    "dbrand",
]
RIDGE_MIN_ROWS = 20


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df else pd.Series(np.nan, index=df.index)


def _pos_log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def _ridge_frame(past: pd.DataFrame, cur: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Признаки строк года t (`cur`, без таргета) по истории `past` (годы < t) и r_{t−1}."""
    prev = past[past["year"] == cur["year"].min() - 1].set_index("department_id")
    ids = cur["department_id"]

    def lag(name: str) -> pd.Series:
        return ids.map(_col(prev, name)) if len(prev) else pd.Series(np.nan, index=cur.index)

    r_prev = lag(TARGET) / lag("active_projects_count").where(lag("active_projects_count") > 0)
    inbound, inbound_prev = _col(cur, "inbound_applications"), lag("inbound_applications")
    dlog_inbound = _pos_log(inbound) - _pos_log(inbound_prev)
    brand = _col(cur, "tech_brand_rating")
    x = pd.DataFrame(
        {
            "dlog_proj": _pos_log(cur["active_projects_count"])
            - _pos_log(lag("active_projects_count")),
            "dlog_staff": _pos_log(_col(cur, "current_staff_count"))
            - _pos_log(lag("current_staff_count")),
            "dlog_inbound": dlog_inbound,
            "inbound_missing": dlog_inbound.isna().astype(float),
            "brand": brand,
            "dbrand": brand - lag("tech_brand_rating"),
        },
        index=cur.index,
    )
    return x[RIDGE_FEATURES].fillna(0.0), r_prev


def predict_ridge(
    history: pd.DataFrame, rows: pd.DataFrame, params_a: RatioParams, ridge_alpha: float = 1.0
) -> pd.Series:
    """Ratio = r_{Y−1} · exp(ẑ), ẑ — Ridge по признакам года Y и Y−1, обученный только на годах
    истории; ẑ ограничен диапазоном z обучения. Без r_{Y−1} или при < RIDGE_MIN_ROWS обучающих
    строк — `predict_ratio` (модель A)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fallback = predict_ratio(history, rows, params_a)  # заодно проверяет утечку и таргет в rows
    xs, zs = [], []
    for t in sorted(history["year"].unique())[1:]:
        cur = history[history["year"] == t]
        x, r_prev = _ridge_frame(history[history["year"] < t], cur.drop(columns=[TARGET]))
        proj = cur["active_projects_count"].where(cur["active_projects_count"] > 0)
        z = np.log(cur[TARGET] / proj) - np.log(r_prev)
        ok = np.isfinite(z)
        xs.append(x[ok])
        zs.append(z[ok])
    x_train = pd.concat(xs) if xs else pd.DataFrame(columns=RIDGE_FEATURES)
    if len(x_train) < RIDGE_MIN_ROWS:
        return fallback
    model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    z_train = pd.concat(zs)
    model.fit(x_train, z_train)
    x, r_prev = _ridge_frame(history, rows)
    z_hat = np.clip(model.predict(x), z_train.min(), z_train.max())  # без экстраполяции
    pred = r_prev * np.exp(z_hat)
    return pred.where(np.isfinite(pred) & (pred > 0)).fillna(fallback).astype(float).rename("ratio")
