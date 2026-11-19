"""T2: smoke-пайплайн (проверка пути и формата, НЕ результат): TF-IDF + LogReg на подвыборке."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from itgorod.config import T2Config
from itgorod.data import build_t2_cache
from itgorod.seed import set_seed


def build_model(cfg: T2Config) -> Pipeline:
    tfidf = {**cfg.tfidf, "ngram_range": tuple(cfg.tfidf.get("ngram_range", (1, 1)))}
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LogisticRegression(**cfg.logreg, random_state=cfg.seed)),
        ]
    )


def run_t2_smoke(cfg: T2Config, force: bool = False) -> Path:
    """Кэш → fit на smoke-подвыборке train → предсказание всего test → smoke-CSV."""
    set_seed(cfg.seed)
    train = pd.read_parquet(build_t2_cache(cfg, "train", smoke=True, force=force))
    test = pd.read_parquet(build_t2_cache(cfg, "test", smoke=True, force=force))
    model = build_model(cfg).fit(train["text"], train["code"])
    pred = pd.DataFrame({"row_id": test["row_id"].astype(str), "code": model.predict(test["text"])})
    sample = pd.read_csv(cfg.paths.sample, dtype=str, usecols=["row_id"])
    out = sample.merge(pred, on="row_id", how="left")
    cfg.paths.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.paths.smoke_output, index=False)
    return cfg.paths.smoke_output


def run_t2_full(cfg: T2Config) -> Path:
    raise NotImplementedError("полный прогон — на этапе T2")
