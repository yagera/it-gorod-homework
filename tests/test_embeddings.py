import inspect

import numpy as np
import pytest

from itgorod import embeddings as emb

MARK = "Текст документа"
DOC = f"РЕШЕНИЕ от 01.02.2025 № 5 Орган А 20.01.2026 {MARK} тело документа номер один"


class FakeEncoder:
    """Детерминированный энкодер без модели: вектор из текста. Записывает вызовы."""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = [(hash((t, j)) % 1000) / 1000 for j in range(self.dim)]
        return out


def _spec(**kw):
    base = dict(model="intfloat/multilingual-e5-base", max_tokens=512, header="with", n_docs=3)
    return emb.EmbeddingSpec(**{**base, **kw})


def test_cache_path_key_has_model_tokens_header_ndocs(tmp_path):
    p = emb.cache_path(tmp_path, _spec())
    name = p.name
    assert "multilingual-e5-base" in name and "/" not in name
    assert "t512" in name and "hwith" in name and "n3" in name and name.endswith(".npy")
    variants = {
        emb.cache_path(tmp_path, _spec(model="BAAI/bge-m3")),
        emb.cache_path(tmp_path, _spec(max_tokens=256)),
        emb.cache_path(tmp_path, _spec(header="none")),
        emb.cache_path(tmp_path, _spec(n_docs=4)),
        p,
    }
    assert len(variants) == 5  # каждый компонент ключа меняет имя файла


def test_cache_hit_does_not_call_encoder_and_roundtrips(tmp_path):
    enc = FakeEncoder()
    texts = [DOC, DOC + " два", "без шапки"]
    a = emb.encode_with_cache(enc, texts, _spec(), tmp_path, batch_size=2)
    assert a.shape == (3, 8) and enc.calls  # первый вызов кодирует
    n_calls = len(enc.calls)
    b = emb.encode_with_cache(enc, texts, _spec(), tmp_path, batch_size=2)
    assert len(enc.calls) == n_calls  # кэш: энкодер не вызывается
    np.testing.assert_array_equal(a, b)
    # другой ключ — другой файл, кодируется заново
    emb.encode_with_cache(enc, texts[:2], _spec(n_docs=2), tmp_path, batch_size=2)
    assert len(enc.calls) > n_calls and len(list(tmp_path.glob("*.npy"))) == 2


def test_cache_rejects_wrong_row_count(tmp_path):
    enc = FakeEncoder()
    emb.encode_with_cache(enc, ["a", "b", "c"], _spec(), tmp_path)
    with pytest.raises(ValueError, match="n_docs"):
        emb.encode_with_cache(enc, ["a", "b"], _spec(), tmp_path)  # spec говорит 3, текстов 2


def test_batches_preserve_order_and_size(tmp_path):
    enc = FakeEncoder()
    texts = [f"док {i}" for i in range(5)]
    out = emb.encode_with_cache(enc, texts, _spec(n_docs=5), tmp_path, batch_size=2)
    assert [len(c) for c in enc.calls] == [2, 2, 1]
    assert [t for c in enc.calls for t in c] == texts
    np.testing.assert_array_equal(out, FakeEncoder().encode(texts).astype(np.float16))


def test_encoder_never_sees_labels(tmp_path):
    """Энкодер получает только тексты: у API кэша и энкодера нет параметров с метками."""
    forbidden = {"y", "labels", "label", "codes", "code", "target", "targets"}
    for fn in (emb.encode_with_cache, emb.benchmark_encoder, emb.texts_for_mode):
        assert not forbidden & set(inspect.signature(fn).parameters), fn.__name__
    assert not forbidden & set(inspect.signature(emb.EmbeddingSpec).parameters)
    enc = FakeEncoder()
    emb.encode_with_cache(enc, [DOC, "x", "y"], _spec(), tmp_path)
    assert all(isinstance(t, str) for call in enc.calls for t in call)
    # ключ кэша от содержимого меток не зависит и не включает их
    assert "010.000.000" not in emb.cache_path(tmp_path, _spec()).name


def test_texts_for_mode():
    with_hdr = emb.texts_for_mode([DOC, "без шапки"], "with")
    none = emb.texts_for_mode([DOC, "без шапки"], "none")
    assert with_hdr == [DOC, "без шапки"]
    assert none == ["тело документа номер один", "без шапки"]  # шапка до маркера убрана
    with pytest.raises(ValueError):
        emb.texts_for_mode([DOC], "bogus")


def test_benchmark_encoder_reports_speed_vram_estimate():
    ticks = iter([0.0, 2.0])  # 2 секунды на кодирование
    enc = FakeEncoder()
    r = emb.benchmark_encoder(
        enc, ["a"] * 10, batch_size=5, clock=lambda: next(ticks), vram_peak_mb=lambda: 1234.0
    )
    assert r["docs_per_s"] == pytest.approx(5.0)
    assert r["peak_vram_mb"] == 1234.0
    assert r["est_50k_min"] == pytest.approx(50000 / 5.0 / 60)
    assert r["n_docs"] == 10


def test_candidates_registry():
    c = emb.CANDIDATES
    assert {"intfloat/multilingual-e5-base", "BAAI/bge-m3", "ai-forever/sbert_large_nlu_ru"} <= set(
        c
    )
    assert c["intfloat/multilingual-e5-base"]["prefix"] == "passage: "
    assert (
        c["BAAI/bge-m3"]["pooling"] == "cls"
        and c["intfloat/multilingual-e5-base"]["pooling"] == "mean"
    )


# ---- кодирование корпуса train/test (с инъекцией энкодера) ----


def _cfg_for_corpus(tmp_path, t2_parquets):
    from itgorod.config import T2Config, T2Paths

    train_p, test_p = t2_parquets
    return T2Config(
        seed=42,
        paths=T2Paths(
            train=train_p,
            test=test_p,
            sample=tmp_path / "s.csv",
            processed_dir=tmp_path / "proc",
            output=tmp_path / "o.csv",
            smoke_output=tmp_path / "so.csv",
        ),
        batch_size=7,
        max_chars=200,
        smoke_n_train_rows=10,
        tfidf={},
        logreg={},
    )


def test_embed_corpus_caches_each_split_and_mode(tmp_path, t2_parquets):
    cfg = _cfg_for_corpus(tmp_path, t2_parquets)
    enc = FakeEncoder()
    cache = tmp_path / "cache"
    paths = emb.embed_corpus(cfg, enc, "intfloat/multilingual-e5-base", 512, cache, batch_size=16)
    names = sorted(p.name for p in paths)
    assert len(names) == 4 and len(set(names)) == 4
    # train: 60 документов, test: 20 (синтетика conftest); режимы with/none
    assert any("hwith_n60" in n for n in names) and any("hnone_n60" in n for n in names)
    assert any("hwith_n20" in n for n in names) and any("hnone_n20" in n for n in names)
    assert all(isinstance(t, str) for call in enc.calls for t in call)
    n_calls = len(enc.calls)
    emb.embed_corpus(cfg, enc, "intfloat/multilingual-e5-base", 512, cache, batch_size=16)
    assert len(enc.calls) == n_calls  # повторный запуск целиком из кэша


def test_embed_corpus_none_mode_strips_header(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from itgorod.config import T2Config, T2Paths

    train = pd.DataFrame(
        {
            "text": [
                f"РЕШЕНИЕ от 01.02.2025 № {i} Орган 20.01.2026 {MARK} тело-{i}" for i in range(6)
            ],
            "code": ["010.000.000"] * 6,
        }
    )
    test = pd.DataFrame(
        {
            "row_id": pd.Series(range(1, 4), dtype="int64"),
            "text": [
                f"ПОСТАНОВЛЕНИЕ от 03.04.2025 № {i} Орган 21.01.2026 {MARK} тест-{i}"
                for i in range(3)
            ],
        }
    )
    tp, sp = tmp_path / "tr.parquet", tmp_path / "te.parquet"
    pq.write_table(pa.Table.from_pandas(train, preserve_index=False), tp)
    pq.write_table(pa.Table.from_pandas(test, preserve_index=False), sp)
    cfg = T2Config(
        seed=42,
        paths=T2Paths(
            train=tp,
            test=sp,
            sample=tmp_path / "s.csv",
            processed_dir=tmp_path / "proc",
            output=tmp_path / "o.csv",
            smoke_output=tmp_path / "so.csv",
        ),
        batch_size=4,
        max_chars=200,
        smoke_n_train_rows=3,
        tfidf={},
        logreg={},
    )
    enc = FakeEncoder()
    emb.embed_corpus(cfg, enc, "intfloat/multilingual-e5-base", 512, tmp_path / "c", batch_size=64)
    calls = enc.calls  # порядок: train/with, train/none, test/with, test/none
    assert len(calls) == 4 and [len(c) for c in calls] == [6, 6, 3, 3]
    assert all(MARK in t for t in calls[0]) and all(MARK in t for t in calls[2])
    assert calls[1] == [f"тело-{i}" for i in range(6)]
    assert calls[3] == [f"тест-{i}" for i in range(3)]
    assert not any("010.000.000" in t for c in calls for t in c)  # меток в тексты не попадает
