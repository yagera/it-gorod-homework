import pandas as pd

from itgorod.config import Fold, T1Config, T1Paths
from itgorod.t1 import run_t1


def test_t1_reproducible(tmp_path, t1_frame):
    train_p, test_p, sample_p = (tmp_path / n for n in ("train.csv", "test.csv", "sample.csv"))
    t1_frame.to_csv(train_p, index=False)
    test = t1_frame[t1_frame["year"] == 2023].drop(columns=["yearly_cv_processed"]).copy()
    test["year"] = 2024
    test.to_csv(test_p, index=False)
    pd.DataFrame({"department_id": test["department_id"], "yearly_cv_processed": 0}).to_csv(
        sample_p, index=False
    )
    outs = []
    for i in range(2):
        cfg = T1Config(
            seed=42,
            paths=T1Paths(train_p, test_p, sample_p, tmp_path / f"out{i}.csv"),
            folds=[Fold(2022, 2023)],
            target_year=2024,
            fallback="tech_stack_median",
        )
        outs.append(run_t1(cfg).read_bytes())
    assert outs[0] == outs[1]
