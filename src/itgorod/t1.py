"""T1: плейсхолдер (НЕ бейзлайн) — значение прошлого года по отделу, чтобы получить валидный CSV."""

from pathlib import Path

import pandas as pd

from itgorod.config import T1Config
from itgorod.data import read_t1
from itgorod.seed import set_seed
from itgorod.validation import assert_features_for_year

TARGET = "yearly_cv_processed"


def predict_placeholder(train: pd.DataFrame, test: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Значение за target_year-1 → последнее известное значение отдела →
    медиана target_year-1 по tech_stack → общая медиана target_year-1."""
    assert_features_for_year(pd.concat([train, test], ignore_index=True), target_year, TARGET)
    hist = train[train["year"] < target_year]
    prev = hist[hist["year"] == target_year - 1]

    by_dept_prev = prev.set_index("department_id")[TARGET]
    last_known = (
        hist.dropna(subset=[TARGET]).sort_values("year").groupby("department_id")[TARGET].last()
    )
    stack_median = prev.groupby("tech_stack")[TARGET].median()
    overall_median = prev[TARGET].median()

    ids = test["department_id"]
    values = ids.map(by_dept_prev)
    values = values.fillna(ids.map(last_known))
    values = values.fillna(test["tech_stack"].map(stack_median))
    values = values.fillna(overall_median)
    return pd.DataFrame({"department_id": ids.to_numpy(), TARGET: values.to_numpy()})


def run_t1(cfg: T1Config) -> Path:
    set_seed(cfg.seed)
    train = read_t1(cfg.paths.train)
    test = read_t1(cfg.paths.test)
    sample = pd.read_csv(cfg.paths.sample, usecols=["department_id"])
    pred = predict_placeholder(train, test, cfg.target_year)
    out = sample.merge(pred, on="department_id", how="left")
    cfg.paths.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.paths.output, index=False)
    return cfg.paths.output
