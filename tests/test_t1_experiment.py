"""Раннер экспериментов T1: сравнение с планкой, общие строки, отсутствие утечки по фолдам."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from itgorod.config import load_t1_config
from itgorod.t1 import BASELINE, TARGET, RatioParams, predict_ridge
from itgorod.t1_experiment import (
    RESULT_COLUMNS,
    compare,
    fold_predictions,
    run_t1_experiment,
    select_params,
)

CFG = load_t1_config("configs/t1.yaml")
FOLDS = CFG.folds
SPECS = {
    "baseline": (BASELINE, False),
    "A": (RatioParams(0.6, 0.5), False),
    "AD": (RatioParams(0.6, 0.5), True),
    "B": lambda h, r: predict_ridge(h, r, RatioParams(0.6, 0.5)) * r["active_projects_count"],
}


@pytest.fixture
def frame() -> pd.DataFrame:
    """20 отделов × 2020–2023: ratio с трендом и шумом, у отдела 3 таргет 2022 — NaN."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(20):
        base, trend = rng.uniform(15, 35), rng.normal(0.2, 0.3)
        for i, year in enumerate(range(2020, 2024)):
            proj = 10.0 + 7 * d + 2 * i
            y = (base + trend * i) * proj * rng.uniform(0.97, 1.03)
            rows.append(
                {
                    "department_id": 1000 + d,
                    "year": year,
                    "tech_stack": "AB"[d % 2],
                    "active_projects_count": proj,
                    "current_staff_count": 2 * proj,
                    "inbound_applications": 50.0 * (d + 1) * (1 + 0.1 * i),
                    "tech_brand_rating": 0.1 * (d % 5),
                    TARGET: np.nan if (d == 3 and year == 2022) else y,
                }
            )
    return pd.DataFrame(rows)


def test_compare_counts():
    y = pd.Series([100.0, 150.0, 1000.0, 2000.0])
    base = pd.Series([110.0, 150.0, 1100.0, 2000.0])  # APE: .10, 0, .10, 0
    pred = pd.Series([105.0, 165.0, 1000.0, 2000.0])  # APE: .05, .10, 0, 0
    r = compare(y, pred, base, small_threshold=200, n_boot=200, seed=0)
    assert (r["n"], r["n_small"], r["n_small_worse"]) == (4, 2, 1)
    assert r["share_improved"] == pytest.approx(0.5)
    assert r["mape"] == pytest.approx(0.0375) and r["mape_small"] == pytest.approx(0.075)
    assert r["delta_mape"] == pytest.approx(0.0375 - 0.05)
    same = compare(y, base, base, 200, 200, 0)
    assert same["delta_mape"] == same["delta_lo"] == same["delta_hi"] == 0


def test_compare_bootstrap_deterministic():
    rng = np.random.default_rng(5)
    y = pd.Series(rng.uniform(50, 500, 40))
    a, b = y * rng.uniform(0.9, 1.1, 40), y * rng.uniform(0.9, 1.1, 40)
    r1, r2 = compare(y, a, b, 200, 300, 7), compare(y, a, b, 200, 300, 7)
    assert (r1["delta_lo"], r1["delta_hi"]) == (r2["delta_lo"], r2["delta_hi"])
    assert r1["delta_lo"] <= r1["delta_mape"] <= r1["delta_hi"]


def test_fold_predictions_common_rows(frame):
    fp = fold_predictions(frame, FOLDS[0], SPECS)  # valid 2022, у отдела 1003 таргет NaN
    assert 1003 not in fp["department_id"].tolist() and len(fp) == 19
    preds = fp[[f"pred_{n}" for n in SPECS]]
    assert np.isfinite(preds.to_numpy()).all()
    assert fp["c_baseline"].eq(1.0).all() and fp["c_A"].eq(1.0).all() and fp["c_B"].eq(1.0).all()


@pytest.mark.parametrize(
    "fold", FOLDS, ids=[f"<={f.train_max_year}->{f.valid_year}" for f in FOLDS]
)
def test_fold_predictions_ignore_valid_and_future(frame, fold):
    """Таргет valid-года (построчно случайный множитель) и все числовые колонки будущих лет
    (включая добавленный 2024) не влияют на прогнозы и множитель c."""
    full = pd.concat([frame, frame[frame.year == 2023].assign(year=2024)], ignore_index=True)
    base = fold_predictions(full, fold, SPECS)
    rng = np.random.default_rng(11)
    mutated = full.copy()
    is_valid = mutated.year == fold.valid_year
    mutated.loc[is_valid, TARGET] *= rng.uniform(0.2, 5.0, is_valid.sum())
    future = mutated.year > fold.valid_year
    nums = [
        c for c in mutated.select_dtypes("number").columns if c not in ("department_id", "year")
    ]
    mutated.loc[future, nums] *= rng.uniform(0.2, 5.0, (future.sum(), len(nums)))
    after = fold_predictions(mutated, fold, SPECS)
    cols = [c for c in base.columns if c.startswith(("pred_", "c_"))]
    pd.testing.assert_frame_equal(base[cols], after[cols], check_exact=True)


def test_fold_predictions_applies_multiplier(frame):
    """A+D = A × c, где c — подобранный множитель (≠ 1 на данных с трендом ratio)."""
    fp = fold_predictions(frame, FOLDS[1], SPECS)
    c = fp["c_AD"].iloc[0]
    assert c != 1.0 and fp["c_AD"].nunique() == 1
    np.testing.assert_allclose(fp["pred_AD"], c * fp["pred_A"])


def test_select_params_returns_grid_point(frame):
    exp = CFG.experiment
    params, grid = select_params(frame, FOLDS, exp)
    assert params.w in exp.w_grid and params.phi in exp.phi_grid
    assert params.outlier_threshold == exp.outlier_threshold
    assert len(grid) == len(exp.w_grid) * len(exp.phi_grid)


def test_run_writes_csv(tmp_path, frame):
    train_p = tmp_path / "train.csv"
    frame.to_csv(train_p, index=False)
    exp = replace(
        CFG.experiment,
        n_bootstrap=50,
        output=tmp_path / "exp.csv",
        grid_output=tmp_path / "grid.csv",
    )
    cfg = replace(CFG, paths=replace(CFG.paths, train=train_p), experiment=exp)
    run_t1_experiment(cfg)
    out = pd.read_csv(exp.output)
    assert list(out.columns) == RESULT_COLUMNS
    assert len(out) == 8 and set(out["model"]) == {"baseline", "A", "A+D", "B"}
    assert (tmp_path / "grid.csv").exists()
