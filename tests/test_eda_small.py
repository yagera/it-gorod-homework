"""Разбор малых значений таргета: функции из itgorod.eda, синтетика, без data/."""

import numpy as np
import pandas as pd
import pytest

from itgorod import eda
from itgorod.config import Fold
from itgorod.t1 import BASELINE, TARGET, predict_target
from itgorod.validation import LeakageError

FOLD = Fold(2022, 2023)


def make(ratios: dict[int, list[float]], projects: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for d, rs in ratios.items():
        for i, r in enumerate(rs):
            p = projects[d][i]
            rows.append(
                {
                    "department_id": 1000 + d,
                    "year": 2020 + i,
                    "tech_stack": "A",
                    "active_projects_count": p,
                    "current_staff_count": 2 * p,
                    TARGET: r * p,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    # d0: обычный; d1: малый таргет в 2023, ratio ушёл 20 → 15 при обвале проектов 40 → 10
    return make(
        {0: [20, 20, 20, 21], 1: [20, 20, 20, 15]},
        {0: [100, 100, 100, 100], 1: [40, 40, 40, 10]},
    )


def test_last_year_rows_columns_and_values(frame):
    rows = eda.last_year_rows(frame, FOLD).set_index("department_id")
    r = rows.loc[1001]
    assert (r["p_year"], r["p_proj"], r["proj"]) == (2022, 40, 10)
    assert r["p_ratio"] == pytest.approx(20) and r["ratio"] == pytest.approx(15)
    assert r["pred"] == pytest.approx(200) and r["y"] == pytest.approx(150)
    assert r["ape"] == pytest.approx(50 / 150)
    assert r["dlp"] == pytest.approx(np.log(10 / 40))
    assert r["dlr"] == pytest.approx(np.log(15 / 20))
    assert rows.loc[1000, "ape"] == pytest.approx(1 / 21)


def test_last_year_rows_pred_matches_baseline_model(frame):
    hist = frame[frame.year < 2023]
    cur = frame[frame.year == 2023].drop(columns=[TARGET])
    model = predict_target(hist, cur, BASELINE).to_numpy()
    rows = eda.last_year_rows(frame, FOLD)
    np.testing.assert_allclose(rows["pred"].to_numpy(), model)


def test_last_year_rows_valid_target_only_in_y_and_ape(frame):
    base = eda.last_year_rows(frame, FOLD)
    mutated = frame.copy()
    mutated.loc[mutated.year >= 2023, TARGET] *= 7
    after = eda.last_year_rows(mutated, FOLD)
    keep = ["p_year", "p_proj", "p_y", "p_ratio", "pred", "proj", "dlp"]
    pd.testing.assert_frame_equal(base[keep], after[keep])
    assert not np.allclose(base["y"], after["y"])


def test_small_rows_threshold(frame):
    small = eda.small_rows(frame, FOLD, threshold=200)
    assert small["department_id"].tolist() == [1001]  # y = 150; у d0 y = 2100


def test_elasticity_by_size_has_all_groups(frame):
    t = eda.elasticity_by_size(frame, max_year=2022, bins=[0, 50, 1e9], n_boot=20, seed=0)
    assert list(t["group"]) == ["(0, 50]", "(50, 1e+09]"]
    assert t["n_depts"].tolist() == [1, 1]


def test_dept_intercepts_detects_offset():
    rows = []
    for d in range(3):
        for i, p in enumerate([10, 20, 30, 40]):
            rows.append(
                {
                    "department_id": d,
                    "year": 2020 + i,
                    "active_projects_count": float(p),
                    TARGET: 500.0 + 2.0 * p + 0.1 * (i % 2),
                }
            )
    t = eda.dept_intercepts(pd.DataFrame(rows))
    assert (t["a_rel"] > 0.5).all() and len(t) == 3


def test_elastic_prediction_equals_baseline_at_alpha_one():
    pure = make({0: [20, 20, 20, 20], 1: [10, 10, 10, 10]}, {0: [50, 60, 70, 80], 1: [5, 6, 7, 8]})
    hist = pure[pure.year < 2023]
    cur = pure[pure.year == 2023].drop(columns=[TARGET])
    base = predict_target(hist, cur, BASELINE).to_numpy()
    np.testing.assert_allclose(eda.predict_elastic(hist, cur).to_numpy(), base, rtol=1e-9)


def test_elastic_prediction_uses_fitted_alpha_and_only_history():
    # y = 3·proj^1.2 → α = 1,2: прогноз при росте проектов выше планки
    rows = []
    rng = np.random.default_rng(0)
    for d in range(30):
        scale = rng.uniform(2, 5)
        for i in range(4):
            p = float(rng.integers(20, 200))
            rows.append(
                {
                    "department_id": d,
                    "year": 2020 + i,
                    "tech_stack": "A",
                    "active_projects_count": p,
                    TARGET: scale * p**1.2,
                }
            )
    df = pd.DataFrame(rows)
    hist, cur = df[df.year < 2023], df[df.year == 2023].drop(columns=[TARGET])
    pred = eda.predict_elastic(hist, cur)
    truth = df[df.year == 2023][TARGET].to_numpy()
    base = predict_target(hist, cur, BASELINE).to_numpy()
    assert np.mean(np.abs(pred - truth) / truth) < 1e-6 < np.mean(np.abs(base - truth) / truth)
    # строки года прогноза в history запрещены
    with pytest.raises(LeakageError):  # строки года прогноза в history запрещены
        eda.predict_elastic(df, cur)


def _two_size_groups() -> pd.DataFrame:
    """Малые отделы (проекты 10–40): y = 5·p^0,8; крупные (150–400): y = 3·p^1,0."""
    rng = np.random.default_rng(3)
    rows = []
    for d in range(60):
        small = d < 30
        scale = rng.uniform(2, 6)
        base = rng.uniform(10, 25) if small else rng.uniform(150, 250)
        for i in range(4):
            p = float(round(base * rng.uniform(0.85, 1.25) * (1 + 0.15 * i)))
            y = scale * p**0.8 if small else scale * p
            rows.append(
                {
                    "department_id": d,
                    "year": 2020 + i,
                    "tech_stack": "A",
                    "active_projects_count": p,
                    TARGET: y,
                }
            )
    return pd.DataFrame(rows)


def test_elastic_grouped_alpha_fits_each_size_group():
    df = _two_size_groups()
    hist, cur = df[df.year < 2023], df[df.year == 2023].drop(columns=[TARGET])
    truth = df[df.year == 2023][TARGET].to_numpy()

    def mape(p):
        return float(np.mean(np.abs(np.asarray(p) - truth) / truth))

    grouped = mape(eda.predict_elastic(hist, cur, bins=[0, 50, 1e9]))
    pooled = mape(eda.predict_elastic(hist, cur))
    base = mape(predict_target(hist, cur, BASELINE))
    assert grouped < 1e-6 < min(pooled, base)


def test_elastic_grouped_unknown_dept_gets_baseline_factor():
    df = _two_size_groups()
    hist = df[df.year < 2023]
    cur = df[df.year == 2023].drop(columns=[TARGET]).copy()
    cur.loc[cur.department_id == 0, "department_id"] = 9999  # нет в истории
    cur.loc[cur.department_id == 9999, "tech_stack"] = "A"
    out = eda.predict_elastic(hist, cur, bins=[0, 50, 1e9])
    base = predict_target(hist, cur, BASELINE)
    i = cur.index[cur.department_id == 9999][0]
    assert out[i] == pytest.approx(base[i]) and np.isfinite(out).all()


def test_elasticity_by_size_ignores_years_after_max_year():
    df = _two_size_groups()
    base = eda.elasticity_by_size(df, max_year=2022, bins=[0, 50, 1e9], n_boot=30, seed=0)
    rng = np.random.default_rng(5)
    mutated = df.copy()
    late = mutated.year > 2022
    mutated.loc[late, TARGET] *= rng.uniform(0.2, 5.0, late.sum())
    mutated.loc[late, "active_projects_count"] *= rng.uniform(0.2, 5.0, late.sum())
    after = eda.elasticity_by_size(mutated, max_year=2022, bins=[0, 50, 1e9], n_boot=30, seed=0)
    pd.testing.assert_frame_equal(base, after)


def test_elastic_p_proj_comes_from_last_year_with_ratio():
    """Если у отдела таргет последнего года истории NaN, p_proj берётся из предыдущего года."""
    df = _two_size_groups()
    df.loc[(df.department_id == 5) & (df.year == 2022), TARGET] = np.nan  # малый отдел, α = 0,8
    hist, cur = df[df.year < 2023], df[df.year == 2023].drop(columns=[TARGET])
    truth = _two_size_groups().query("department_id == 5 and year == 2023")[TARGET].iloc[0]
    out = eda.predict_elastic(hist, cur, bins=[0, 50, 1e9])
    assert out[cur.index[cur.department_id == 5][0]] == pytest.approx(truth, rel=1e-6)


def test_elastic_zero_projects_gives_finite_prediction():
    df = _two_size_groups()
    hist, cur = df[df.year < 2023], df[df.year == 2023].drop(columns=[TARGET]).copy()
    cur.loc[cur.department_id == 3, "active_projects_count"] = 0.0
    assert np.isfinite(eda.predict_elastic(hist, cur, bins=[0, 50, 1e9])).all()
