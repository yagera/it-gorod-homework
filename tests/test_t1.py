import pandas as pd
import pytest

from itgorod.config import Fold, T1Config, T1Paths
from itgorod.data import read_t1
from itgorod.t1 import predict_placeholder, run_t1
from itgorod.validation import LeakageError, read_submission_csv, validate_t1_submission

TARGET = "yearly_cv_processed"


@pytest.fixture
def t1_test(t1_frame) -> pd.DataFrame:
    """Test 2024: признаки без таргета, порядок отделов перемешан."""
    last = t1_frame[t1_frame["year"] == 2023].drop(columns=[TARGET]).copy()
    last["year"] = 2024
    return last.iloc[::-1].reset_index(drop=True)


def _pred(train, test) -> pd.Series:
    out = predict_placeholder(train, test, 2024)
    return out.set_index("department_id")[TARGET]


def test_read_t1_drops_empty_row(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "department_id,year,tech_stack,yearly_cv_processed\n1000000,2020,Парк,5.0\n,,,\n",
        encoding="utf-8",
    )
    df = read_t1(path)
    assert len(df) == 1
    assert df["year"].dtype.kind == "i" and df["department_id"].dtype.kind == "i"


def test_placeholder_uses_last_year(t1_frame, t1_test):
    t1_frame.loc[(t1_frame.department_id == 1000005) & (t1_frame.year == 2023), TARGET] = 777.0
    assert _pred(t1_frame, t1_test)[1000005] == 777.0


def test_placeholder_fallback_last_known(t1_frame, t1_test):
    # dept 1000002: NaN в 2023, в 2022 = 1000 + 200 + 20
    assert _pred(t1_frame, t1_test)[1000002] == 1220.0


def test_placeholder_fallback_stack_median(t1_frame, t1_test):
    # dept 1000001 без таргетов вообще, стек Метро → медиана 2023 по стеку Метро
    metro_2023 = t1_frame[(t1_frame.tech_stack == "Метро") & (t1_frame.year == 2023)][TARGET]
    pred = predict_placeholder(t1_frame, t1_test, 2024)
    assert pred.set_index("department_id")[TARGET][1000001] == metro_2023.median()
    assert pred[TARGET].notna().all() and (pred[TARGET] > 0).all()


def test_placeholder_keeps_test_order(t1_frame, t1_test):
    pred = predict_placeholder(t1_frame, t1_test, 2024)
    assert pred["department_id"].tolist() == t1_test["department_id"].tolist()


def test_placeholder_rejects_future(t1_frame, t1_test):
    future = t1_frame[t1_frame.year == 2023].copy()
    future["year"] = 2025
    with pytest.raises(LeakageError):
        predict_placeholder(pd.concat([t1_frame, future]), t1_test, 2024)


def test_placeholder_deterministic(t1_frame, t1_test):
    pd.testing.assert_frame_equal(
        predict_placeholder(t1_frame, t1_test, 2024), predict_placeholder(t1_frame, t1_test, 2024)
    )


def test_run_t1_writes_valid_csv(tmp_path, t1_frame, t1_test):
    train_p, test_p, sample_p = (tmp_path / n for n in ("train.csv", "test.csv", "sample.csv"))
    t1_frame.to_csv(train_p, index=False)
    t1_test.to_csv(test_p, index=False)
    sample = pd.DataFrame({"department_id": sorted(t1_test["department_id"]), TARGET: 0})
    sample.to_csv(sample_p, index=False)
    out = tmp_path / "nested" / "out" / "task-1.csv"  # папки нет — должна создаться
    cfg = T1Config(
        seed=42,
        paths=T1Paths(train=train_p, test=test_p, sample=sample_p, output=out),
        folds=[Fold(2022, 2023)],
        target_year=2024,
        fallback="tech_stack_median",
    )
    assert run_t1(cfg) == out
    validate_t1_submission(read_submission_csv(out), read_submission_csv(sample_p))
