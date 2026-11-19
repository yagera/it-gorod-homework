import pandas as pd
import pytest

from itgorod.config import T2Config, T2Paths
from itgorod.data import read_t2_codes
from itgorod.t2 import run_t2_full, run_t2_smoke
from itgorod.validation import read_submission_csv, validate_t2_submission


def _cfg(tmp_path, train_p, test_p, n=40) -> T2Config:
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"row_id": range(1, 21), "code": "000.000.000"}).to_csv(sample, index=False)
    return T2Config(
        seed=42,
        paths=T2Paths(
            train=train_p,
            test=test_p,
            sample=sample,
            processed_dir=tmp_path / "processed",
            output=tmp_path / "out" / "task-2.csv",
            smoke_output=tmp_path / "out" / "task-2.smoke.csv",  # папки нет
        ),
        batch_size=7,
        max_chars=200,
        smoke_n_train_rows=n,
        tfidf={"max_features": 1000, "ngram_range": [1, 2], "sublinear_tf": True},
        logreg={"C": 4.0, "max_iter": 200, "class_weight": "balanced"},
    )


def _check(cfg, train_p) -> pd.DataFrame:
    out = read_submission_csv(cfg.paths.smoke_output)
    validate_t2_submission(out, read_submission_csv(cfg.paths.sample), read_t2_codes(train_p))
    return out


def test_t2_smoke_writes_valid_csv(tmp_path, t2_parquets):
    cfg = _cfg(tmp_path, *t2_parquets)
    assert run_t2_smoke(cfg) == cfg.paths.smoke_output
    out = _check(cfg, t2_parquets[0])
    assert out["code"].str.fullmatch(r"[0-9]{3}\.[0-9]{3}\.[0-9]{3}").all()
    assert out["code"].isin({"010.000.000", "020.150.030", "290.000.000"}).all()


def test_t2_smoke_subset_missing_class(tmp_path, t2_parquets):
    # первые 2 строки train — два разных класса; третий класс в подвыборку не попадает
    cfg = _cfg(tmp_path, *t2_parquets, n=2)
    run_t2_smoke(cfg)
    out = _check(cfg, t2_parquets[0])
    assert out["code"].nunique() <= 2


def test_t2_full_writes_valid_csv(tmp_path, t2_parquets):
    cfg = _cfg(tmp_path, *t2_parquets)
    assert run_t2_full(cfg) == cfg.paths.output
    out = read_submission_csv(cfg.paths.output)
    sample = read_submission_csv(cfg.paths.sample)
    validate_t2_submission(out, sample, read_t2_codes(t2_parquets[0]))
    assert len(out) == len(sample) == 20
    assert out["row_id"].tolist() == sample["row_id"].tolist()  # порядок как в sample
    assert out["code"].map(type).eq(str).all()
    assert out["code"].str.fullmatch(r"[0-9]{3}\.[0-9]{3}\.[0-9]{3}").all()
    assert set(out["code"]) <= {"010.000.000", "020.150.030", "290.000.000"}
    assert out["code"].str.startswith("0").any()  # ведущий нуль не потерян


def _unique_parquets(tmp_path):
    """Тексты train и test заведомо различны (маркеры TRAIN-/TEST-), чтобы fit на test был виден."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    codes = ["010.000.000", "020.150.030", "290.000.000"]
    train = pd.DataFrame(
        {
            "text": [f"TRAIN-{i} {codes[i % 3][:3]}x " * 30 for i in range(60)],
            "code": [codes[i % 3] for i in range(60)],
        }
    )
    test = pd.DataFrame(
        {
            "row_id": pd.Series(range(1, 21), dtype="int64"),
            "text": [f"TEST-{i} {codes[i % 3][:3]}x " * 30 for i in range(20)],
        }
    )
    tp, sp = tmp_path / "tr.parquet", tmp_path / "te.parquet"
    pq.write_table(pa.Table.from_pandas(train, preserve_index=False), tp)
    pq.write_table(pa.Table.from_pandas(test, preserve_index=False), sp)
    return tp, sp, train


def _run_full_with_spies(tmp_path, monkeypatch):
    from spies import Calls, make_spies

    from itgorod import t2 as t2_mod

    calls = Calls()
    spy_vec, spy_model = make_spies(calls)
    monkeypatch.setattr(t2_mod, "make_vectorizer", lambda cfg: spy_vec())
    monkeypatch.setattr(t2_mod, "make_model", lambda exp, seed: spy_model())
    train_p, test_p, train = _unique_parquets(tmp_path)
    run_t2_full(_cfg(tmp_path, train_p, test_p))
    return calls, train


def test_t2_full_fit_only_on_train(tmp_path, monkeypatch):
    """Правило A: любой fit векторайзера и модели — ровно на train; test не попадает ни в один."""
    calls, train = _run_full_with_spies(tmp_path, monkeypatch)
    train_texts = set(train["text"].str.slice(0, 2000))
    assert len(calls.vec_fits) == 1 and len(calls.model_fits) == 1
    for docs in calls.vec_fits:
        assert len(docs) == 60 and set(docs) == train_texts
        assert not any(d.startswith("TEST-") for d in docs)
    n_rows, labels = calls.model_fits[0]
    assert n_rows == 60 and labels == train["code"].tolist()


def test_t2_full_rejects_row_id_mismatch(tmp_path, t2_parquets):
    cfg = _cfg(tmp_path, *t2_parquets)
    pd.DataFrame({"row_id": range(100, 120), "code": "000.000.000"}).to_csv(
        cfg.paths.sample, index=False
    )
    with pytest.raises(ValueError, match="row_id"):
        run_t2_full(cfg)


def test_t2_smoke_reproducible(tmp_path, t2_parquets):
    cfg = _cfg(tmp_path, *t2_parquets)
    run_t2_smoke(cfg, force=True)
    first = cfg.paths.smoke_output.read_bytes()
    run_t2_smoke(cfg, force=True)
    assert cfg.paths.smoke_output.read_bytes() == first
