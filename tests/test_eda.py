import numpy as np
import pandas as pd
import pytest

from itgorod import eda
from itgorod.config import Fold

FOLDS = [Fold(2021, 2022), Fold(2022, 2023)]


def _frame() -> pd.DataFrame:
    rows = []
    for d in range(6):
        for y in range(2020, 2024):
            proj = 10.0 + 5 * d + y - 2020
            rows.append(
                {
                    "department_id": 1000 + d,
                    "year": y,
                    "tech_stack": "A" if d % 2 else "B",
                    "yearly_cv_processed": (20.0 + d) * proj,
                    "active_projects_count": proj,
                    "current_staff_count": 2 * proj,
                    "avg_dev_salary": 100.0,
                    "total_salary_budget": 1.2 * 2 * proj * 100.0,
                    "devs_hired": 3.0,
                    "devs_fired": 2.0,
                    "turnover_rate": 5.0 / (2 * proj),
                    "inbound_applications": 7.0 * proj,
                    "share_seniors": 10.0,
                    "tech_brand_rating": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_quality_report_counts_empty_dups_and_outliers():
    df = _frame()
    df.loc[0, "inbound_applications"] = 0
    df.loc[1, "turnover_rate"] = 1.5
    raw = pd.concat([df, df.iloc[[2]], pd.DataFrame([[np.nan] * df.shape[1]], columns=df.columns)])
    q = eda.quality_report(raw)
    assert q["n_fully_empty"] == 1
    assert q["n_dup_dept_year"] == 1
    assert q["n_inbound_zero"] == 1
    assert q["n_turnover_gt1"] == 1


def test_ratio_is_constant_inside_department():
    rs = eda.ratio_stability(_frame())
    assert rs["cv_within_dept"]["max"] < 1e-9
    assert rs["n_rows_dev_gt15pct"] == 0


def test_ratio_model_exact_on_synthetic_and_no_leak_on_fold():
    diag = eda.ratio_model_diagnostics(_frame(), FOLDS)
    assert (diag.loc["ratio_dept_mean"] < 1e-9).all()
    assert (diag.loc["lag_target"] > 0).all()
    assert (diag.loc["n_rows"] == 6).all()


def test_predictions_ignore_valid_and_future_targets():
    rng = np.random.default_rng(0)
    df = _frame()
    df[eda.TARGET] *= rng.uniform(0.9, 1.1, len(df))  # ratio шумит по годам
    fold = FOLDS[0]
    _, base = eda.fold_predictions(df, fold)
    mutated = df.copy()
    mutated.loc[mutated["year"] >= fold.valid_year, eda.TARGET] *= 5
    _, after = eda.fold_predictions(mutated, fold)
    for name in base:
        pd.testing.assert_series_equal(base[name], after[name])


def test_identities_constant():
    ident = eda.identities(_frame())
    assert np.allclose(ident["mean"], [1.2, 1.0])
    assert (ident["std"] < 1e-9).all()


def test_drift_table_detects_shift():
    df = _frame()
    test = df[df["year"] == 2023].assign(year=2024).drop(columns=[eda.TARGET])
    test["avg_dev_salary"] = 200.0
    dr = eda.drift_table(df, test, 2023)
    assert dr.index[0] == "avg_dev_salary"
    assert dr.loc["avg_dev_salary", "shift_pct"] == 100.0


def test_stack_constancy():
    df = _frame()
    test = df[df["year"] == 2023].assign(year=2024)
    sc = eda.stack_constancy(df, test)
    assert sc["n_depts_multi_stack"] == 0 and sc["test_matches_train"] == 1.0


def _elastic_frame(alpha: float, n_depts: int = 30, noise: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for d in range(n_depts):
        r = rng.uniform(10, 40)
        for y in range(2020, 2024):
            proj = rng.uniform(5, 300)
            rows.append(
                {
                    "department_id": d,
                    "year": y,
                    "active_projects_count": proj,
                    eda.TARGET: r * proj**alpha * np.exp(rng.normal(0, noise)),
                }
            )
    return pd.DataFrame(rows)


def test_elasticity_recovers_alpha():
    res = eda.projects_elasticity(_elastic_frame(1.2), n_boot=300, seed=0)
    assert abs(res["alpha"] - 1.2) < 0.02
    assert res["ci_lo"] < 1.2 < res["ci_hi"]
    assert res["n_depts"] == 30 and res["n_rows"] == 120


def test_elasticity_one_for_pure_ratio():
    assert eda.projects_elasticity(_frame(), n_boot=50, seed=0)["alpha"] == pytest.approx(
        1, abs=1e-9
    )


def test_elasticity_deterministic():
    df = _elastic_frame(0.9)
    assert eda.projects_elasticity(df, 100, 3) == eda.projects_elasticity(df, 100, 3)
