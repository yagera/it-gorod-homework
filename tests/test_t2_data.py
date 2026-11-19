import os

import pandas as pd

from itgorod.config import T2Config, T2Paths
from itgorod.data import build_t2_cache, cache_path, iter_t2_batches, read_t2_codes


def _cfg(tmp_path, train_p, test_p, max_chars=100, n=40) -> T2Config:
    return T2Config(
        seed=42,
        paths=T2Paths(
            train=train_p,
            test=test_p,
            sample=tmp_path / "sample.csv",
            processed_dir=tmp_path / "processed" / "deep",  # папки нет
            output=tmp_path / "out" / "task-2.csv",
            smoke_output=tmp_path / "out" / "task-2.smoke.csv",
        ),
        batch_size=7,
        max_chars=max_chars,
        smoke_n_train_rows=n,
        tfidf={},
        logreg={},
    )


def test_batches_truncate(t2_parquets):
    train_p, _ = t2_parquets
    frames = list(iter_t2_batches(train_p, ["text", "code"], batch_size=7, max_chars=100))
    df = pd.concat(frames)
    assert len(df) == 60 and len(frames) > 1
    assert df["text"].str.len().max() <= 100
    assert df["code"].map(lambda c: isinstance(c, str)).all()
    assert "010.000.000" in set(df["code"])  # ведущий ноль сохранён


def test_batches_limit(t2_parquets):
    train_p, _ = t2_parquets
    frames = iter_t2_batches(train_p, ["text", "code"], batch_size=10, max_chars=100, limit=25)
    assert sum(len(f) for f in frames) == 25


def test_cache_created_and_reused(tmp_path, t2_parquets):
    cfg = _cfg(tmp_path, *t2_parquets)
    path = build_t2_cache(cfg, "train", smoke=True)
    assert path.exists() and len(pd.read_parquet(path)) == 40
    os.utime(path, (1, 1))
    assert build_t2_cache(cfg, "train", smoke=True) == path
    assert path.stat().st_mtime == 1  # не перезаписан
    build_t2_cache(cfg, "train", smoke=True, force=True)
    assert path.stat().st_mtime > 1


def test_test_cache_is_full_even_in_smoke(tmp_path, t2_parquets):
    cfg = _cfg(tmp_path, *t2_parquets, n=5)
    df = pd.read_parquet(build_t2_cache(cfg, "test", smoke=True))
    assert len(df) == 20 and list(df.columns) == ["row_id", "text"]


def test_cache_path_depends_on_params(tmp_path, t2_parquets):
    a = cache_path(_cfg(tmp_path, *t2_parquets, max_chars=100), "train", True)
    b = cache_path(_cfg(tmp_path, *t2_parquets, max_chars=200), "train", True)
    c = cache_path(_cfg(tmp_path, *t2_parquets, n=10), "train", True)
    assert len({a, b, c}) == 3
    t_a = cache_path(_cfg(tmp_path, *t2_parquets, n=10), "test", True)
    t_b = cache_path(_cfg(tmp_path, *t2_parquets, n=20), "test", False)
    assert t_a == t_b


def test_read_t2_codes(t2_parquets, t2_codes):
    assert read_t2_codes(t2_parquets[0]) == t2_codes
