import pandas as pd
import pytest

from itgorod.validation import (
    SubmissionError,
    read_submission_csv,
    validate_t1_submission,
    validate_t2_submission,
)


def _t1_ok(sample: pd.DataFrame) -> pd.DataFrame:
    sub = sample.copy()
    sub["yearly_cv_processed"] = [str(1000 + i) for i in range(len(sub))]
    return sub


def _t2_ok(sample: pd.DataFrame, codes: set[str]) -> pd.DataFrame:
    sub = sample.copy()
    sub["code"] = [sorted(codes)[i % len(codes)] for i in range(len(sub))]
    return sub


def test_t1_valid_passes(t1_sample):
    validate_t1_submission(_t1_ok(t1_sample), t1_sample)


def _extra_col(s):
    s.insert(0, "Unnamed: 0", range(len(s)))
    return s


def _drop_row(s):
    return s.iloc[:-1]


def _swap_ids(s):
    s.loc[[0, 1], "department_id"] = s.loc[[1, 0], "department_id"].to_numpy()
    return s


def _set(value):
    def f(s):
        s.loc[3, "yearly_cv_processed"] = value
        return s

    return f


@pytest.mark.parametrize(
    "mutate",
    [_extra_col, _drop_row, _swap_ids, _set("0"), _set("-5"), _set(""), _set("inf")],
    ids=["extra_col", "drop_row", "swap_ids", "zero", "negative", "empty", "inf"],
)
def test_t1_rejects(t1_sample, mutate):
    sub = mutate(_t1_ok(t1_sample))
    with pytest.raises(SubmissionError):
        validate_t1_submission(sub, t1_sample)


def test_t1_collects_all_errors(t1_sample):
    sub = _set("0")(_swap_ids(_t1_ok(t1_sample)))
    with pytest.raises(SubmissionError) as exc:
        validate_t1_submission(sub, t1_sample)
    assert len(exc.value.errors) >= 2


def test_t2_valid_passes(t2_sample, t2_codes):
    validate_t2_submission(_t2_ok(t2_sample, t2_codes), t2_sample, t2_codes)


def _code(value):
    def f(s):
        s.loc[2, "code"] = value
        return s

    return f


def _t2_extra_col(s):
    s["extra"] = "x"
    return s


def _t2_swap_rows(s):
    s.loc[[0, 1], "row_id"] = s.loc[[1, 0], "row_id"].to_numpy()
    return s


@pytest.mark.parametrize(
    "mutate",
    [
        _code("10.0.0"),
        _code("10.000.000"),
        _code("999.000.000"),
        _code("010000000"),
        _code(""),
        _t2_extra_col,
        _t2_swap_rows,
    ],
    ids=["short", "lost_zero", "unknown_class", "no_dots", "empty", "extra_col", "row_order"],
)
def test_t2_rejects(t2_sample, t2_codes, mutate):
    sub = mutate(_t2_ok(t2_sample, t2_codes))
    with pytest.raises(SubmissionError):
        validate_t2_submission(sub, t2_sample, t2_codes)


def test_t2_roundtrip_keeps_zeros(tmp_path, t2_sample, t2_codes):
    sub = _t2_ok(t2_sample, t2_codes)
    path = tmp_path / "sub.csv"
    sub.to_csv(path, index=False)
    back = read_submission_csv(path)
    validate_t2_submission(back, t2_sample, t2_codes)
    assert back["code"].tolist() == sub["code"].tolist()


@pytest.mark.parametrize(
    "code", ["０１０.000.000", "٠١٠.000.000"], ids=["fullwidth", "arabic_indic"]
)
def test_t2_rejects_unicode_digits_even_if_allowed(t2_sample, t2_codes, code):
    allowed = t2_codes | {code}
    sub = _t2_ok(t2_sample, t2_codes)
    sub.loc[2, "code"] = code
    with pytest.raises(SubmissionError, match="NNN.NNN.NNN"):
        validate_t2_submission(sub, t2_sample, allowed)
