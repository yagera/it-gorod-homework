import dataclasses
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from itgorod.config import T2Config, T2Paths
from itgorod.data import (
    HEADER_EXTRA,
    TAIL_SEP,
    build_t2_cache,
    cache_path,
    iter_t2_batches,
    read_t2_codes,
)


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


# ---- режимы «голова + хвост» и «без шапки» (чтение на уровне Arrow) ----

MARK = "Текст документа"


def _ref(t: str, head: int, tail: int, strip: bool) -> str:
    """Эталон на полном тексте: без шапки (если strip), затем голова + хвост без перекрытия."""
    if strip:
        i = t[: head + HEADER_EXTRA].find(MARK)
        if i >= 0:
            t = t[i + len(MARK) :].lstrip()
    if tail == 0:
        return t[:head]
    return t if len(t) <= head + tail else t[:head] + TAIL_SEP + t[-tail:]


def _docs():
    hdr = f"ПОСТАНОВЛЕНИЕ от 01.02.2025 № 5 Орган А 20.01.2026 {MARK} "
    body = "".join(chr(0x430 + i % 32) for i in range(5000))  # кириллица, 5000 символов
    out = []
    for n in (0, 10, 149, 150, 151, 160, 400, 1500, 3000, 5000):
        out.append(hdr + body[:n])  # с шапкой
        out.append(body[:n])  # без маркера
    return out


@pytest.fixture
def doc_parquet(tmp_path):
    texts = _docs()
    path = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": texts, "code": ["010.000.000"] * len(texts)}), path)
    return path, texts


@pytest.mark.parametrize("head,tail,strip", [(100, 50, False), (100, 0, True), (100, 50, True)])
def test_head_tail_strip_matches_reference(doc_parquet, head, tail, strip):
    path, texts = doc_parquet
    df = pd.concat(
        iter_t2_batches(
            path,
            ["text", "code"],
            batch_size=4,
            max_chars=head,
            tail_chars=tail,
            strip_header=strip,
        )
    )
    assert df["text"].tolist() == [_ref(t, head, tail, strip) for t in texts]
    assert df["code"].map(type).eq(str).all()


def test_head_only_default_unchanged(doc_parquet):
    path, texts = doc_parquet
    df = pd.concat(iter_t2_batches(path, ["text"], batch_size=4, max_chars=100))
    assert df["text"].tolist() == [t[:100] for t in texts]


def test_tail_has_no_overlap_with_head(doc_parquet):
    path, texts = doc_parquet
    df = pd.concat(iter_t2_batches(path, ["text"], batch_size=4, max_chars=100, tail_chars=50))
    for raw, got in zip(texts, df["text"], strict=True):
        assert len(got) <= 100 + len(TAIL_SEP) + 50 or got == raw
        if len(raw) <= 150:
            assert got == raw  # короткий документ — целиком, без дублей


def test_cache_path_depends_on_tail_and_strip(tmp_path, t2_parquets):
    base = _cfg(tmp_path, *t2_parquets)
    paths = {
        cache_path(base, "train", False),
        cache_path(dataclasses.replace(base, tail_chars=50), "train", False),
        cache_path(dataclasses.replace(base, strip_header=True), "train", False),
    }
    assert len(paths) == 3
    assert cache_path(base, "train", False).name == f"t2_train_c{base.max_chars}.parquet"
