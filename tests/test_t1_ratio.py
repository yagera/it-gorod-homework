"""Модель A (сглаживание ratio отдела) и множитель D. Синтетика, без data/."""

import numpy as np
import pandas as pd
import pytest

from itgorod.t1 import (
    BASELINE,
    TARGET,
    RatioParams,
    fit_multiplier,
    predict_ratio,
    predict_ridge,
    predict_target,
    weighted_median,
)
from itgorod.validation import LeakageError


def make_frame(
    ratios: dict[int, list[float]], stacks: dict[int, str] | None = None
) -> pd.DataFrame:
    """Отдел d: ratio по годам 2020.. из списка, проекты = 10 + 5d + (year − 2020)."""
    rows = []
    for d, rs in ratios.items():
        for i, r in enumerate(rs):
            proj = 10.0 + 5 * d + i
            rows.append(
                {
                    "department_id": 1000 + d,
                    "year": 2020 + i,
                    "tech_stack": (stacks or {}).get(d, "A" if d % 2 else "B"),
                    "active_projects_count": proj,
                    TARGET: r * proj,
                }
            )
    return pd.DataFrame(rows)


def rows_for(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["year"] == year].drop(columns=[TARGET])


def _ratio(history, rows, params) -> pd.Series:
    return predict_ratio(history, rows, params).set_axis(rows["department_id"].to_numpy())


def test_baseline_equals_last_year_ratio():
    df = make_frame({0: [20, 21, 22, 30], 1: [10, 11, 12, 13]})
    hist, rows = df[df.year < 2023], rows_for(df, 2023)
    pred = predict_target(hist, rows, BASELINE)
    assert np.allclose(
        pred.to_numpy(),
        [22 * rows.iloc[0].active_projects_count, 12 * rows.iloc[1].active_projects_count],
    )


def test_blend_and_trend():
    df = make_frame({0: [20, 21, 22, 0]})
    hist, rows = df[df.year < 2023], rows_for(df, 2023)
    assert _ratio(hist, rows, RatioParams(w=1, phi=1))[1000] == pytest.approx(23)
    assert _ratio(hist, rows, RatioParams(w=0, phi=0))[1000] == pytest.approx(21)
    assert _ratio(hist, rows, RatioParams(w=0.5, phi=0))[1000] == pytest.approx(21.5)


def test_outlier_year_excluded():
    df = make_frame({0: [20, 20, 30, 0]})
    hist, rows = df[df.year < 2023], rows_for(df, 2023)
    assert _ratio(hist, rows, RatioParams(w=1, phi=0))[1000] == pytest.approx(20)


def test_all_outliers_fall_back_to_raw_last():
    df = make_frame({0: [10, 20, 0]})
    hist, rows = df[df.year < 2022], rows_for(df, 2022)
    assert _ratio(hist, rows, RatioParams(w=1, phi=0))[1000] == pytest.approx(20)


def test_nonpositive_trend_falls_back():
    df = make_frame({0: [9, 3, 0]})
    hist, rows = df[df.year < 2022], rows_for(df, 2022)
    params = RatioParams(w=1, phi=1, outlier_threshold=float("inf"))
    assert _ratio(hist, rows, params)[1000] == pytest.approx(3)


def test_unknown_dept_uses_stack_then_overall_median():
    df = make_frame({0: [10, 10, 0], 1: [20, 20, 0], 3: [30, 30, 0]})  # стеки B, A, A
    hist = df[df.year < 2022]
    rows = pd.DataFrame(
        {
            "department_id": [2000, 2001],
            "year": [2022, 2022],
            "tech_stack": ["A", "C"],
            "active_projects_count": [10.0, 10.0],
        }
    )
    r = _ratio(hist, rows, RatioParams(w=1, phi=0))
    assert r[2000] == pytest.approx(25)  # медиана стека A: 20, 20, 30, 30
    assert r[2001] == pytest.approx(20)  # общая медиана: 10,10,20,20,30,30


def test_rows_alignment():
    df = make_frame({0: [20, 20, 0], 1: [10, 10, 0], 2: [30, 30, 0]})
    hist = df[df.year < 2022]
    rows = rows_for(df, 2022).iloc[[2, 0, 1]].set_axis([100, 7, 42])
    r = predict_ratio(hist, rows, RatioParams(w=1, phi=0))
    assert r.index.equals(rows.index)
    assert r.tolist() == pytest.approx([30, 20, 10])


def test_nan_projects_gives_nan_target():
    df = make_frame({0: [20, 20, 0], 1: [10, 10, 0]})
    hist, rows = df[df.year < 2022], rows_for(df, 2022).copy()
    rows.loc[rows.department_id == 1000, "active_projects_count"] = np.nan
    pred = predict_target(hist, rows, BASELINE).set_axis(rows["department_id"].to_numpy())
    assert np.isnan(pred[1000]) and np.isfinite(pred[1001])


def test_guards():
    df = make_frame({0: [20, 20, 20]})
    with pytest.raises(LeakageError):
        predict_ratio(df, rows_for(df, 2022), BASELINE)
    with pytest.raises(ValueError):
        predict_ratio(df[df.year < 2022], df[df.year == 2022], BASELINE)


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3
    assert weighted_median(np.array([2.0, 1.0]), np.array([1.0, 1.0])) == 1


def test_multiplier_recovers_scale():
    df = make_frame({d: [10 * (d + 1) * 1.05**i for i in range(4)] for d in range(4)})
    assert fit_multiplier(df, BASELINE) == pytest.approx(1.05, abs=1e-9)


def test_multiplier_one_year_history():
    df = make_frame({0: [20], 1: [10]})
    assert fit_multiplier(df, BASELINE) == 1.0


def test_multiplier_scales_prediction_and_reduces_error():
    """Ratio растёт на 5% в год: c ≈ 1,05, прогноз масштабируется ровно на c и становится точным."""
    df = make_frame({d: [10 * (d + 1) * 1.05**i for i in range(4)] for d in range(4)})
    hist, rows = df[df.year < 2023], rows_for(df, 2023)
    c = fit_multiplier(hist, BASELINE)
    plain = predict_target(hist, rows, BASELINE)
    scaled = predict_target(hist, rows, BASELINE, c)
    truth = df[df.year == 2023][TARGET].to_numpy()
    assert c == pytest.approx(1.05, abs=1e-9)
    assert np.allclose(scaled.to_numpy(), c * plain.to_numpy())
    mape_plain = np.mean(np.abs(plain.to_numpy() - truth) / truth)
    mape_scaled = np.mean(np.abs(scaled.to_numpy() - truth) / truth)
    assert mape_plain > 0.04 and mape_scaled < 1e-9


def test_predict_ratio_rejects_empty_history():
    df = make_frame({0: [20, 20, 20]})
    empty = df[df.year < 2020]
    with pytest.raises(ValueError, match="history is empty"):
        predict_ratio(empty, rows_for(df, 2022), BASELINE)
    with pytest.raises(ValueError, match="history is empty"):
        predict_ridge(empty, rows_for(df, 2022), BASELINE)


def make_signal_frame(n_depts: int = 60, beta: float = 0.5) -> pd.DataFrame:
    """ratio_t = ratio_{t−1} · exp(beta · dlog_inbound + шум 0,001)."""
    rng = np.random.default_rng(2)
    rows = []
    for d in range(n_depts):
        r, inbound = rng.uniform(15, 35), rng.uniform(200, 2000)
        for i, year in enumerate(range(2020, 2024)):
            if i:
                new_inbound = inbound * np.exp(rng.normal(0, 0.3))
                r *= np.exp(beta * np.log(new_inbound / inbound) + rng.normal(0, 0.001))
                inbound = new_inbound
            proj = 20.0 + d + 3 * i
            rows.append(
                {
                    "department_id": 1000 + d,
                    "year": year,
                    "tech_stack": "AB"[d % 2],
                    "active_projects_count": proj,
                    "current_staff_count": 2 * proj,
                    "inbound_applications": inbound,
                    "tech_brand_rating": 0.5,
                    TARGET: r * proj,
                }
            )
    return pd.DataFrame(rows)


def test_ridge_learns_signal():
    df = make_signal_frame()
    hist, rows = df[df.year < 2023], rows_for(df, 2023)
    true = (df[df.year == 2023][TARGET] / df[df.year == 2023]["active_projects_count"]).to_numpy()
    ridge = predict_ridge(hist, rows, BASELINE).to_numpy()
    last = predict_ratio(hist, rows, BASELINE).to_numpy()
    assert np.mean(np.abs(ridge - true) / true) < 0.005
    assert np.mean(np.abs(last - true) / true) > 0.05  # сигнал есть, планка его не видит


def test_ridge_falls_back_without_last_ratio():
    df = make_signal_frame()
    hist = df[df.year < 2023].copy()
    hist.loc[(hist.department_id == 1000) & (hist.year == 2022), TARGET] = np.nan
    rows = rows_for(df, 2023)
    ridge = predict_ridge(hist, rows, BASELINE).set_axis(rows["department_id"].to_numpy())
    a = _ratio(hist, rows, BASELINE)
    assert ridge[1000] == pytest.approx(a[1000])


def test_ridge_guards_and_alignment():
    df = make_signal_frame()
    hist, rows = df[df.year < 2023], rows_for(df, 2023)
    shuffled = rows.sample(frac=1, random_state=0).set_axis(range(500, 500 + len(rows)))
    r1 = predict_ridge(hist, rows, BASELINE).to_numpy()
    r2 = predict_ridge(hist, shuffled, BASELINE)
    assert r2.index.equals(shuffled.index)
    pd.testing.assert_series_equal(
        r2.set_axis(shuffled["department_id"].to_numpy()).sort_index(),
        pd.Series(r1, index=rows["department_id"].to_numpy(), name=r2.name).sort_index(),
    )
    with pytest.raises(LeakageError):
        predict_ridge(df, rows, BASELINE)
    with pytest.raises(ValueError):
        predict_ridge(hist, df[df.year == 2023], BASELINE)


def test_ridge_prediction_within_training_z_range():
    """Экстремальный признак на прогнозе не уводит ratio за диапазон изменений, виденных в
    обучении (без экстраполяции и переполнения exp)."""
    df = make_signal_frame()
    hist, rows = df[df.year < 2023], rows_for(df, 2023).copy()
    rows["inbound_applications"] *= 1e6
    ratio = predict_ridge(hist, rows, RatioParams(0.7, 0.25)).to_numpy()
    h = hist.assign(r=hist[TARGET] / hist["active_projects_count"]).sort_values("year")
    z = np.log(h.groupby("department_id")["r"].pct_change().dropna() + 1)
    r_prev = h[h.year == 2022].set_index("department_id")["r"].reindex(rows["department_id"])
    dz = np.log(ratio / r_prev.to_numpy())
    assert np.isfinite(ratio).all()
    assert dz.max() <= z.max() + 1e-9 and dz.min() >= z.min() - 1e-9
