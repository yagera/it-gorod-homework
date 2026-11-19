"""Проверки сабмишнов (и далее — утечек по времени). Используются pytest, CLI, скиллами и CI."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CODE_RE = re.compile(r"[0-9]{3}\.[0-9]{3}\.[0-9]{3}")
T1_COLUMNS = ["department_id", "yearly_cv_processed"]
T2_COLUMNS = ["row_id", "code"]


class SubmissionError(ValueError):
    """Сабмишн не соответствует формату; `errors` — все найденные нарушения."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def read_submission_csv(path: str | Path) -> pd.DataFrame:
    """Читает сабмишн строго как строки (иначе теряются ведущие нули в `code`)."""
    return pd.read_csv(path, dtype=str)


def _check_frame(sub: pd.DataFrame, sample: pd.DataFrame, columns: list[str], id_col: str):
    errors: list[str] = []
    if list(sub.columns) != columns:
        errors.append(f"columns must be exactly {columns}, got {list(sub.columns)}")
        return errors
    if len(sub) != len(sample):
        errors.append(f"expected {len(sample)} rows, got {len(sub)}")
    elif not (sub[id_col].astype(str).to_numpy() == sample[id_col].astype(str).to_numpy()).all():
        errors.append(f"{id_col} values/order differ from sample_submission")
    if sub[id_col].duplicated().any():
        errors.append(f"duplicate {id_col} values")
    return errors


def validate_t1_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    errors = _check_frame(sub, sample, T1_COLUMNS, "department_id")
    if list(sub.columns) == T1_COLUMNS:
        values = pd.to_numeric(sub["yearly_cv_processed"], errors="coerce")
        bad = ~np.isfinite(values) | (values <= 0)
        if bad.any():
            errors.append(
                f"yearly_cv_processed must be finite and > 0; {int(bad.sum())} bad value(s), "
                f"first at row {int(np.flatnonzero(bad.to_numpy())[0])}"
            )
    if errors:
        raise SubmissionError(errors)


def validate_t2_submission(
    sub: pd.DataFrame, sample: pd.DataFrame, allowed_codes: set[str]
) -> None:
    errors = _check_frame(sub, sample, T2_COLUMNS, "row_id")
    if list(sub.columns) == T2_COLUMNS:
        codes = sub["code"]
        bad_fmt = ~codes.map(lambda c: isinstance(c, str) and bool(CODE_RE.fullmatch(c)))
        if bad_fmt.any():
            first = codes[bad_fmt].iloc[0]
            errors.append(f"{int(bad_fmt.sum())} code(s) not in NNN.NNN.NNN format, e.g. {first!r}")
        unknown = ~bad_fmt & ~codes.isin(allowed_codes)
        if unknown.any():
            example = codes[unknown].iloc[0]
            errors.append(f"{int(unknown.sum())} code(s) not present in train, e.g. {example!r}")
    if errors:
        raise SubmissionError(errors)
