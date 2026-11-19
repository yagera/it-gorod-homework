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
    """Правило A на месте вызова: любой fit векторайзера и модели видит ровно train-часть
    фолда (документы и метки); valid не попадает ни в один fit."""
    from spies import Calls, make_spies

    cfg = _cfg(tmp_path)
    calls = Calls()
    spy_vec, spy_model = make_spies(calls)
    monkeypatch.setattr(te, "make_vectorizer", lambda cfg: spy_vec())
    monkeypatch.setattr(te, "make_model", lambda exp, seed: spy_model())
    te.run_experiments(
        cfg,
        [te.Experiment("svc", "none", 5000)],
        tmp_path / "e.csv",
        tmp_path / "pc.csv",
        log=lambda *_: None,
    )
    meta = te.load_meta(cfg)
    texts = te.load_train(cfg, 5000)["text"].to_numpy()
    y = meta["code"].to_numpy()
    schemes = te.make_schemes(meta, cfg.seed)
    expected = [
        (frozenset(texts[folds != k]), y[folds != k].tolist())
        for folds in schemes.values()
        for k in range(te.N_SPLITS)
    ]
    assert len(calls.vec_fits) == len(calls.model_fits) == len(expected)
    for docs, (n_rows, labels), (train_docs, train_labels) in zip(
        calls.vec_fits, calls.model_fits, expected, strict=True
    ):
        assert len(docs) == len(train_docs) and frozenset(docs) == train_docs
        assert n_rows == len(train_docs) and labels == train_labels


def test_experiment_ids_and_variants():
    base = te.Experiment("svc", "none", 2000)
    assert base.id == "t2-svc-none-c2000"  # прежние id не меняются
    assert te.Experiment("svc", "none", 1500, tail_chars=500).id == "t2-svc-none-c1500-t500"
    assert te.Experiment("svc", "none", 2000, strip_header=True).id == "t2-svc-none-c2000-nohdr"
    assert te.Experiment("svc", "none", 2000, C=0.25).id == "t2-svc-none-c2000-C0.25"
    assert (
        base.variant == (2000, 0, False)
        and te.Experiment("svc", "none", 2000, C=2).variant == base.variant
    )
    assert te.make_model(te.Experiment("svc", "none", 2000, C=2.0), 42).C == 2.0


def test_run_experiments_variants_main_only_and_append(tmp_path):
    cfg = _cfg(tmp_path)
    out, pc = tmp_path / "e.csv", tmp_path / "pc.csv"
    g1 = [te.Experiment("svc", "none", 5000)]
    te.run_experiments(cfg, g1, out, pc, log=lambda *_: None)  # обе схемы
    g2 = [
        te.Experiment("svc", "none", 600, tail_chars=200),
        te.Experiment("svc", "none", 600, tail_chars=200, C=2.0),  # те же признаки, другое C
        te.Experiment("svc", "none", 600, strip_header=True),
    ]
    res = te.run_experiments(
        cfg, g2, out, pc, log=lambda *_: None, scheme_names=("main",), append=True
    )
    assert "control_macro_f1" not in res.columns and len(res) == 3
    allrows = pd.read_csv(out)
    assert list(allrows["id"]) == [
        "t2-svc-none-c5000",
        "t2-svc-none-c600-t200",
        "t2-svc-none-c600-t200-C2",
        "t2-svc-none-c600-nohdr",
    ]
    assert allrows.loc[0, "tail_chars"] == 0 and allrows.loc[1, "tail_chars"] == 200
    assert allrows["strip_header"].tolist() == [False, False, False, True]
    assert allrows.loc[0, "control_macro_f1"] > 0 and pd.isna(allrows.loc[1, "control_macro_f1"])
    pcd = pd.read_csv(pc, dtype={"code": str})
    assert set(pcd["id"]) == set(allrows["id"]) and pcd["code"].str.len().eq(11).all()
    # повторный запуск того же id заменяет строку, а не дублирует
    te.run_experiments(
        cfg, g2[:1], out, pc, log=lambda *_: None, scheme_names=("main",), append=True
    )
    assert pd.read_csv(out)["id"].tolist().count("t2-svc-none-c600-t200") == 1


def test_variant_fit_only_on_train(tmp_path, monkeypatch):
    """Правило A для вариантов head+tail и без шапки: те же гарантии, что для базового."""
    from spies import Calls, make_spies

    cfg = _cfg(tmp_path)
    calls = Calls()
    spy_vec, spy_model = make_spies(calls)
    monkeypatch.setattr(te, "make_vectorizer", lambda cfg: spy_vec())
    monkeypatch.setattr(te, "make_model", lambda exp, seed: spy_model())
    exp = te.Experiment("svc", "none", 600, tail_chars=200)
    te.run_experiments(
        cfg,
        [exp],
        tmp_path / "e.csv",
        tmp_path / "pc.csv",
        log=lambda *_: None,
        scheme_names=("main",),
    )
    meta = te.load_meta(cfg)
    texts = te.load_train(cfg, 600, 200)["text"].to_numpy()
    folds = te.make_schemes(meta, cfg.seed)["main"]
    assert len(calls.vec_fits) == len(calls.model_fits) == te.N_SPLITS
    y = meta["code"].to_numpy()
    for k, (docs, (n_rows, labels)) in enumerate(
        zip(calls.vec_fits, calls.model_fits, strict=True)
    ):
        assert frozenset(docs) == frozenset(texts[folds != k]) and len(docs) == (folds != k).sum()
        assert n_rows == len(docs) and labels == y[folds != k].tolist()


def test_append_to_legacy_csv_without_new_columns(tmp_path):
    """Старый CSV сетки (без tail_chars/strip_header/C) дополняется без потери строк."""
    cfg = _cfg(tmp_path)
    out, pc = tmp_path / "e.csv", tmp_path / "pc.csv"
    te.run_experiments(cfg, [te.Experiment("svc", "none", 5000)], out, pc, log=lambda *_: None)
    legacy = pd.read_csv(out).drop(columns=["tail_chars", "strip_header", "C"])
    legacy.to_csv(out, index=False)
    te.run_experiments(
        cfg,
        [te.Experiment("svc", "none", 600, tail_chars=200)],
        out,
        pc,
        log=lambda *_: None,
        scheme_names=("main",),
        append=True,
    )
    df = pd.read_csv(out)
    assert df.loc[0, "id"] == "t2-svc-none-c5000"
    assert (df.loc[0, "tail_chars"], bool(df.loc[0, "strip_header"]), df.loc[0, "C"]) == (
        0,
        False,
        0.5,
    )
    assert df["tail_chars"].tolist() == [0, 200] and df["tail_chars"].dtype.kind == "i"
