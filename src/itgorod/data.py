"""Чтение данных: T1 (CSV) и, далее, T2 (parquet батчами)."""

from pathlib import Path

import pandas as pd


def read_t1(path: str | Path) -> pd.DataFrame:
    """CSV T1; полностью пустые строки выбрасываются, department_id и year → int."""
    df = pd.read_csv(path).dropna(how="all")
    return df.astype({"department_id": "int64", "year": "int64"})
