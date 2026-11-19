import pandas as pd
import pytest

from itgorod.config import Fold, load_t1_config
from itgorod.validation import (
    LeakageError,
    assert_features_for_year,
    assert_folds_ordered,
    assert_no_time_leakage,
    split_fold,
)

FOLDS = load_t1_config("configs/t1.yaml").folds


@pytest.mark.parametrize(
    "fold", FOLDS, ids=[f"<={f.train_max_year}->{f.valid_year}" for f in FOLDS]
)
def test_fold_has_no_leakage(t1_frame, fold):
    train, valid = split_fold(t1_frame, fold)
    assert_no_time_leakage(train, valid)
    assert valid["year"].unique().tolist() == [fold.valid_year]
    assert train["year"].max() == fold.train_max_year


def test_folds_ordered():
    assert_folds_ordered(FOLDS)
    with pytest.raises(LeakageError):
        assert_folds_ordered([Fold(2022, 2023), Fold(2021, 2022)])
    with pytest.raises(LeakageError):
        assert_folds_ordered([Fold(2022, 2022)])


def test_overlap_detected(t1_frame):
    train = t1_frame[t1_frame["year"] <= 2023]
    valid = t1_frame[t1_frame["year"] == 2023]
    with pytest.raises(LeakageError):
        assert_no_time_leakage(train, valid)


def _with_year(frame: pd.DataFrame, year: int, target: float) -> pd.DataFrame:
    row = frame[frame["year"] == 2023].iloc[[0]].copy()
    row["year"] = year
    row["yearly_cv_processed"] = target
    return pd.concat([frame, row], ignore_index=True)


def test_features_future_year_rejected(t1_frame):
    with pytest.raises(LeakageError):
        assert_features_for_year(
            _with_year(t1_frame, 2025, float("nan")), 2024, "yearly_cv_processed"
        )


def test_features_target_of_year_rejected(t1_frame):
    with pytest.raises(LeakageError):
        assert_features_for_year(_with_year(t1_frame, 2024, 5.0), 2024, "yearly_cv_processed")


def test_features_of_year_without_target_pass(t1_frame):
    ok = _with_year(t1_frame, 2024, float("nan"))
    assert_features_for_year(ok, 2024, "yearly_cv_processed")
