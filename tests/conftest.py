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
