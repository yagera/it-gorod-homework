import pandas as pd
import pytest


@pytest.fixture
def t1_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department_id": [str(1000000 + i) for i in range(10)],
            "yearly_cv_processed": ["0"] * 10,
        }
    )


@pytest.fixture
def t2_sample() -> pd.DataFrame:
    return pd.DataFrame({"row_id": [str(i) for i in range(1, 21)], "code": ["000.000.000"] * 20})


@pytest.fixture
def t2_codes() -> set[str]:
    return {"010.000.000", "020.150.030", "290.000.000"}


@pytest.fixture
def t1_frame() -> pd.DataFrame:
    """10 отделов × 2020–2023; у dept 1 все таргеты NaN, у dept 2 NaN в 2023."""
    stacks = ["Парк", "Метро", "Пляж"]
    rows = []
    for d in range(10):
        for year in range(2020, 2024):
            target = 1000.0 + 100 * d + 10 * (year - 2020)
            if d == 1 or (d == 2 and year == 2023):
                target = float("nan")
            rows.append(
                {
                    "department_id": 1000000 + d,
                    "year": year,
                    "tech_stack": stacks[d % 3],
                    "yearly_cv_processed": target,
                    "current_staff_count": 100 + d,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def t2_parquets(tmp_path, t2_codes):
    """Синтетический T2: train 60 строк (каждый код по 20), test 20 строк."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    codes = sorted(t2_codes)
    train = pd.DataFrame(
        {
            "text": [f"ДОК {codes[i % 3]} " + ("слово " * (10 + 700 * (i % 5))) for i in range(60)],
            "code": [codes[i % 3] for i in range(60)],
        }
    )
    test = pd.DataFrame(
        {
            "row_id": pd.Series(range(1, 21), dtype="int64"),
            "text": [f"ДОК {codes[i % 3]} " + ("слово " * (10 + 700 * (i % 5))) for i in range(20)],
        }
    )
    train_p, test_p = tmp_path / "train.parquet", tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pandas(train, preserve_index=False), train_p)
    pq.write_table(pa.Table.from_pandas(test, preserve_index=False), test_p)
    return train_p, test_p
