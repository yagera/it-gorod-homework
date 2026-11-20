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
        base.variant == (2000, 0, False, "")
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


# ---- абляция шапки по частям ----


def test_header_mode_ids_and_variant():
    e = te.Experiment("svc", "none", 600, strip_header=True, C=2.0, header_mode="only_org")
    assert e.id == "t2-svc-none-c600-nohdr-C2-hdr-only_org"
    assert e.variant != te.Experiment("svc", "none", 600, strip_header=True, C=2.0).variant
    assert te.header_mode_parts("only_type") == ("type",)
    assert te.header_mode_parts("no_org") == ("type", "numdates")
    assert te.header_mode_parts("full") == ("type", "org", "numdates")
    with pytest.raises(ValueError):
        te.header_mode_parts("bogus")


def test_load_variant_texts_parts(tmp_path):
    cfg = _cfg(tmp_path)

    def texts(mode):
        exp = te.Experiment("svc", "none", 600, strip_header=True, header_mode=mode)
        return te.load_variant_texts(cfg, exp)

    body = te.load_train(cfg, 600, 0, True)["text"].to_numpy()
    only_org, no_org, only_type, full = (
        texts("only_org"),
        texts("no_org"),
        texts("only_type"),
        texts("full"),
    )
    assert only_org[3] == "Орган 3 " + body[3]
    assert only_type[3] == "РЕШЕНИЕ " + body[3]
    assert "Орган 3" not in no_org[3] and no_org[3].startswith(
        "РЕШЕНИЕ от 01.02.2025 № 3 20.01.2026"
    )
    assert full[3].startswith("РЕШЕНИЕ Орган 3 от 01.02.2025 № 3 20.01.2026 ")
    assert len(full) == len(body) == 90


def test_header_variant_fit_only_on_train(tmp_path, monkeypatch):
    from spies import Calls, make_spies

    cfg = _cfg(tmp_path)
    calls = Calls()
    spy_vec, spy_model = make_spies(calls)
    monkeypatch.setattr(te, "make_vectorizer", lambda cfg: spy_vec())
    monkeypatch.setattr(te, "make_model", lambda exp, seed: spy_model())
    exp = te.Experiment("svc", "none", 600, strip_header=True, header_mode="only_org")
    te.run_experiments(cfg, [exp], tmp_path / "e.csv", tmp_path / "pc.csv", log=lambda *_: None)
    meta = te.load_meta(cfg)
    texts = te.load_variant_texts(cfg, exp)
    schemes = te.make_schemes(meta, cfg.seed)
    expected = [frozenset(texts[f != k]) for f in schemes.values() for k in range(te.N_SPLITS)]
    assert len(calls.vec_fits) == len(calls.model_fits) == len(expected)
    for docs, train_docs in zip(calls.vec_fits, expected, strict=True):
        assert frozenset(docs) == train_docs


def test_header_parse_rate(tmp_path):
    cfg = _cfg(tmp_path)
    r = te.header_parse_rate(cfg)
    assert r["n"] == 90 and r["parsed"] == 90 and r["share"] == 1.0


def test_extra_scheme_folds_and_rule_a(tmp_path, monkeypatch):
    """Дополнительная схема (группы по дате): число фолдов из массива, fit — только вне фолда."""
    from spies import Calls, make_spies

    cfg = _cfg(tmp_path)
    n = 90
    folds = np.arange(n) % 3  # 3 группы
    calls = Calls()
    spy_vec, spy_model = make_spies(calls)
    monkeypatch.setattr(te, "make_vectorizer", lambda cfg: spy_vec())
    monkeypatch.setattr(te, "make_model", lambda exp, seed: spy_model())
    exp = te.Experiment("svc", "none", 5000)
    res = te.run_experiments(
        cfg,
        [exp],
        tmp_path / "e.csv",
        tmp_path / "pc.csv",
        log=lambda *_: None,
        scheme_names=("date",),
        extra_schemes={"date": folds},
    )
    assert "date_macro_f1" in res.columns and "main_macro_f1" not in res.columns
    texts = te.load_train(cfg, 5000)["text"].to_numpy()
    assert len(calls.vec_fits) == len(calls.model_fits) == 3
    for k, docs in enumerate(calls.vec_fits):
        assert frozenset(docs) == frozenset(texts[folds != k]) and len(docs) == (folds != k).sum()


def test_append_merges_schemes_for_same_id(tmp_path):
    """Дозапись другой схемы к существующему id не стирает ранее посчитанные колонки."""
    cfg = _cfg(tmp_path)
    out, pc = tmp_path / "e.csv", tmp_path / "pc.csv"
    exp = te.Experiment("svc", "none", 5000)
    te.run_experiments(cfg, [exp], out, pc, log=lambda *_: None)  # main + control
    before = pd.read_csv(out).iloc[0]
    folds = np.arange(90) % 4  # классы циклятся по 3: каждый класс есть в train-части фолда
    te.run_experiments(
        cfg,
        [exp],
        out,
        pc,
        log=lambda *_: None,
        scheme_names=("date",),
        extra_schemes={"date": folds},
        append=True,
    )
    after = pd.read_csv(out)
    assert len(after) == 1
    row = after.iloc[0]
    assert row["main_macro_f1"] == before["main_macro_f1"]
    assert row["control_macro_f1"] == before["control_macro_f1"]
    assert row["date_macro_f1"] > 0
    # per-class: строки всех трёх схем на месте
    assert set(pd.read_csv(pc)["scheme"]) == {"main", "control", "date"}


def test_duplicate_ids_in_grid_run_once(tmp_path):
    cfg = _cfg(tmp_path)
    e = te.Experiment("svc", "none", 5000, C=2.0)
    res = te.run_experiments(
        cfg, [e, e], tmp_path / "e.csv", tmp_path / "pc.csv", log=lambda *_: None
    )
    assert len(res) == 1 and len(pd.read_csv(tmp_path / "e.csv")) == 1


# ---- оценка с подменой даты выгрузки в valid ----


def _dated_cfg(tmp_path):
    """Как `_cfg`, но у документов разные даты выгрузки (19..23 января)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(0)
    codes = ["010.000.000", "020.000.000", "030.000.000"]
    words = {c: [f"{c[:3]}{w}" for w in "abcdefgh"] for c in codes}
    texts, ys = [], []
    for i in range(90):
        c = codes[i % 3]
        body = " ".join(rng.choice(words[c], 80)) + f" уник{i}"
        d = 19 + i % 5
        texts.append(
            f"РЕШЕНИЕ от 01.02.2025 № {i} Орган {i % 30} {d}.01.2026 Текст документа {body}"
        )
        ys.append(c)
    pq.write_table(pa.table({"text": texts, "code": ys}), cfg.paths.train)
    return cfg


def test_swap_experiment_id_and_validation():
    e = te.Experiment("svc", "none", 2000, C=1.0, perturb="swap_fetch_date")
    assert e.id == "t2-svc-none-c2000-C1-swapdate"
    assert e.variant == te.Experiment("svc", "none", 2000, C=1.0).variant  # те же тексты
    assert e.group_key != te.Experiment("svc", "none", 2000, C=1.0).group_key
    with pytest.raises(ValueError):
        te.Experiment("svc", "none", 2000, perturb="bogus").id  # noqa: B018


def test_swap_touches_only_valid_not_train(tmp_path, monkeypatch):
    from spies import Calls, make_spies

    cfg = _dated_cfg(tmp_path)
    calls = Calls()
    spy_vec, spy_model = make_spies(calls)
    monkeypatch.setattr(te, "make_vectorizer", lambda cfg: spy_vec())
    monkeypatch.setattr(te, "make_model", lambda exp, seed: spy_model())
    pools = []
    real_swap = te.swap_valid_dates

    def recording_swap(valid, pool, seed):
        pools.append((list(pool), seed))
        return real_swap(valid, pool, seed)

    monkeypatch.setattr(te, "swap_valid_dates", recording_swap)
    exp = te.Experiment("svc", "none", 5000, perturb="swap_fetch_date")
    res = te.run_experiments(
        cfg,
        [exp],
        tmp_path / "e.csv",
        tmp_path / "pc.csv",
        log=lambda *_: None,
        scheme_names=("main",),
    )
    meta = te.load_meta(cfg)
    texts = te.load_train(cfg, 5000)["text"].to_numpy()
    folds = te.make_schemes(meta, cfg.seed)["main"]
    assert len(calls.vec_fits) == len(calls.vec_transforms) == te.N_SPLITS
    from itgorod.t2_recon import FETCH_RE

    n_diff = 0
    for k in range(te.N_SPLITS):
        # train-часть: fit ровно на исходных текстах (подмена train не трогает)
        assert calls.vec_fits[k] == list(texts[folds != k])
        # valid: отличается только датой выгрузки
        orig, got = list(texts[folds == k]), calls.vec_transforms[k]
        assert len(got) == len(orig)
        for o, g in zip(orig, got, strict=True):
            mo, mg = FETCH_RE.search(o), FETCH_RE.search(g)
            assert o[: mo.start("fetch")] == g[: mg.start("fetch")]
            assert o[mo.end("fetch") :] == g[mg.end("fetch") :]
            n_diff += o != g
        # даты берутся из train-части фолда
        pool = {FETCH_RE.search(t).group("fetch") for t in texts[folds != k]}
        assert {FETCH_RE.search(g).group("fetch") for g in got} <= pool
    # пул дат — ровно даты train-части своего фолда (с кратностями), seed фиксирован
    from itgorod.t2_cv import fetch_date_pool

    assert len(pools) == te.N_SPLITS
    for k, (pool, seed) in enumerate(pools):
        assert sorted(pool) == sorted(fetch_date_pool(list(texts[folds != k])))
        assert tuple(seed) == (cfg.seed, k)
    assert n_diff > 0 and res.loc[0, "main_n_perturbed"] == n_diff
