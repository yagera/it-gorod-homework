import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from itgorod import t2_experiment as te
from itgorod.config import T2Config, T2Paths


def _cfg(tmp_path) -> T2Config:
    rng = np.random.default_rng(0)
    codes = ["010.000.000", "020.000.000", "030.000.000"]
    words = {c: [f"{c[:3]}{w}" for w in "abcdefgh"] for c in codes}
    texts, ys = [], []
    for i in range(90):
        c = codes[i % 3]
        body = " ".join(rng.choice(words[c], 80)) + f" уник{i}"
        texts.append(
            f"РЕШЕНИЕ от 01.02.2025 № {i} Орган {i % 30} 20.01.2026 Текст документа {body}"
        )
        ys.append(c)
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"text": texts, "code": ys}), path)
    return T2Config(
        seed=42,
        paths=T2Paths(
            train=path,
            test=path,
            sample=tmp_path / "s.csv",
            processed_dir=tmp_path / "proc",
            output=tmp_path / "o.csv",
            smoke_output=tmp_path / "so.csv",
        ),
        batch_size=20,
        max_chars=5000,
        smoke_n_train_rows=10,
        tfidf={"max_features": 500, "ngram_range": [1, 1]},
        logreg={},
    )


def test_grid_ids_unique_and_cover_models():
    g = te.default_grid()
    assert len({e.id for e in g}) == len(g)
    assert {e.model for e in g} == {"logreg", "svc", "cnb"}
    assert {e.max_chars for e in g} == {2000, 5000}
    assert [e.id for e in g if e.class_weight == "balanced"] == ["t2-svc-balanced-c5000"]


def test_run_experiments_synthetic(tmp_path):
    cfg = _cfg(tmp_path)
    grid = [te.Experiment("svc", "balanced", 5000), te.Experiment("cnb", "none", 5000)]
    out = te.run_experiments(
        cfg, grid, tmp_path / "e.csv", tmp_path / "pc.csv", log=lambda *_: None
    )
    assert list(out["id"]) == [e.id for e in grid]
    assert (out["main_macro_f1"] > 0.9).all() and (out["control_macro_f1"] > 0.9).all()
    pc = pd.read_csv(tmp_path / "pc.csv", dtype={"code": str})
    assert set(pc["scheme"]) == {"main", "control"} and pc["code"].str.len().eq(11).all()
    # в контрольной схеме органы не пересекаются между train и valid → все строки «без органа»
    assert (out["control_n_unseen"] == 90).all()


def test_memory_guard(monkeypatch):
    monkeypatch.setattr(te, "peak_rss_mb", lambda: 5000.0)
    monkeypatch.setattr(te, "MEMORY_LIMIT_MB", 4096.0)
    with pytest.raises(te.MemoryLimitExceeded):
        te.check_memory()


def test_run_experiments_fit_only_on_train_part(tmp_path, monkeypatch):
    """Правило A на месте вызова: векторайзер и модель видят только train-часть фолда."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    cfg = _cfg(tmp_path)
    fit_docs, fit_labels = [], []

    class SpyVec(TfidfVectorizer):
        def fit_transform(self, raw_documents, y=None):
            fit_docs.append(list(raw_documents))
            return super().fit_transform(raw_documents, y)

        def fit(self, raw_documents, y=None):
            fit_docs.append(list(raw_documents))
            return super().fit(raw_documents, y)

    class SpyModel(LinearSVC):
        def fit(self, X, y, **kw):
            fit_labels.append((X.shape[0], len(y)))
            return super().fit(X, y, **kw)

    monkeypatch.setattr(te, "make_vectorizer", lambda cfg: SpyVec())
    monkeypatch.setattr(te, "make_model", lambda exp, seed: SpyModel())
    te.run_experiments(
        cfg,
        [te.Experiment("svc", "none", 5000)],
        tmp_path / "e.csv",
        tmp_path / "pc.csv",
        log=lambda *_: None,
    )
    meta = te.load_meta(cfg)
    texts = te.load_train(cfg, 5000)["text"].to_numpy()
    schemes = te.make_schemes(meta, cfg.seed)
    expected = {
        frozenset(texts[folds != k]) for folds in schemes.values() for k in range(te.N_SPLITS)
    }
    assert len(fit_docs) == 2 * te.N_SPLITS
    assert {frozenset(d) for d in fit_docs} == expected
    for (n_rows, n_labels), docs in zip(fit_labels, fit_docs, strict=True):
        assert n_rows == n_labels == len(docs)  # метки только train-части
