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


def test_t2_full_fit_never_sees_test_texts(tmp_path, t2_parquets, monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer

    from itgorod import t2 as t2_mod

    seen = []

    class SpyVec(TfidfVectorizer):
        def fit_transform(self, raw_documents, y=None):
            seen.append(list(raw_documents))
            return super().fit_transform(raw_documents, y)

    monkeypatch.setattr(t2_mod, "make_vectorizer", lambda cfg: SpyVec())
    cfg = _cfg(tmp_path, *t2_parquets)
    run_t2_full(cfg)
    # fit ровно один раз и ровно на 60 документах train (в test их 20)
    assert len(seen) == 1 and len(seen[0]) == 60


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
