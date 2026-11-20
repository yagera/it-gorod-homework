"""T2: smoke-пайплайн (проверка пути и формата, НЕ результат): TF-IDF + LogReg на подвыборке."""

import dataclasses
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from itgorod.config import T2Config
from itgorod.data import build_t2_cache
from itgorod.seed import set_seed
from itgorod.t2_experiment import Experiment, make_model, make_vectorizer


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


FULL_MAX_CHARS = 2000
# финалист настройки (записи 16, 17): LinearSVC без весов, 2000 символов, C=1 — наименьший C
# в пределах σ от лучшего; фабрики признаков и модели общие с экспериментами
FULL_EXPERIMENT = Experiment("svc", "none", FULL_MAX_CHARS, C=1.0)


def run_t2_full(cfg: T2Config, force: bool = False) -> Path:
    """Сабмишн финалиста: LinearSVC без весов (C=1), word 30k + char 50k, `max_chars=2000`.
    Векторайзер и модель обучаются на всём train; test только `transform`/`predict`."""
    set_seed(cfg.seed)
    full = dataclasses.replace(
        cfg, max_chars=FULL_MAX_CHARS, tail_chars=0, strip_header=False
    )  # как в эксперименте финалиста
    train = pd.read_parquet(build_t2_cache(full, "train", smoke=False, force=force))
    test = pd.read_parquet(build_t2_cache(full, "test", smoke=False, force=force))
    vec = make_vectorizer(cfg)
    x_train = vec.fit_transform(train["text"])
    model = make_model(FULL_EXPERIMENT, cfg.seed).fit(x_train, train["code"])
    pred = pd.DataFrame(
        {
            "row_id": test["row_id"].astype(str),
            "code": model.predict(vec.transform(test["text"])).astype(str),
        }
    )
    sample = pd.read_csv(cfg.paths.sample, dtype=str, usecols=["row_id"])
    out = sample.merge(pred, on="row_id", how="left")
    if out["code"].isna().any() or len(out) != len(sample) or len(pred) != len(sample):
        raise ValueError("row_id test и sample не совпадают: у части строк sample нет прогноза")
    cfg.paths.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.paths.output, index=False)
    return cfg.paths.output
