import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from itgorod import t2_embed_eval as ee


def _toy(n=120, k=3, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["010.000.000", "020.000.000", "030.000.000"])[np.arange(n) % k]
    centers = rng.normal(size=(k, d)) * 3
    X = centers[np.arange(n) % k] + rng.normal(size=(n, d))
    folds = np.arange(n) % 4
    return X.astype(np.float32), y, folds


def test_zscore_rows():
    z = ee.zscore_rows(np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]]))
    assert z[0].mean() == pytest.approx(0) and z[0].std() == pytest.approx(1)
    assert np.isfinite(z[1]).all() and (z[1] == 0).all()  # постоянная строка не даёт NaN


def test_blend_scores_weights():
    a = np.array([[3.0, 2.0, 0.0]])
    b = np.array([[0.0, 2.0, 3.0]])
    classes = np.array(["a", "b", "c"])
    assert ee.blend_predict(a, b, 0.0, classes)[0] == "a"  # только первая модель
    assert ee.blend_predict(a, b, 1.0, classes)[0] == "c"  # только вторая
    assert ee.blend_predict(a, b, 0.5, classes)[0] == "b"  # поровну: средний класс


def test_oof_logits_fit_only_on_train_part(monkeypatch):
    X, y, folds = _toy()
    seen_scaler, seen_lr = [], []
    real_fit_s, real_fit_l = StandardScaler.fit, LogisticRegression.fit

    def spy_s(self, X_, *a, **k):
        seen_scaler.append(len(X_))
        return real_fit_s(self, X_, *a, **k)

    def spy_l(self, X_, y_, *a, **k):
        seen_lr.append((len(X_), len(y_)))
        return real_fit_l(self, X_, y_, *a, **k)

    monkeypatch.setattr(StandardScaler, "fit", spy_s)
    monkeypatch.setattr(LogisticRegression, "fit", spy_l)
    scores, classes = ee.oof_logits(X, y, folds, C=1.0, max_iter=200)
    n_train = [int((folds != k).sum()) for k in range(4)]
    assert seen_scaler == n_train  # scaler видел только train-часть каждого фолда
    assert seen_lr == [(n, n) for n in n_train]
    assert scores.shape == (len(y), len(classes)) and list(classes) == sorted(set(y))
    assert (classes[scores.argmax(1)] == y).mean() > 0.9  # игрушка разделима


def test_place_scores_floor_for_missing_class():
    fold_scores = np.array([[2.0, -1.0], [0.5, 3.0]])  # модель знает два класса из трёх
    out = ee.place_scores(fold_scores, ["a", "c"], np.array(["a", "b", "c"]))
    assert out[:, 0].tolist() == [2.0, 0.5] and out[:, 2].tolist() == [-1.0, 3.0]
    # отсутствующий класс b: минимум строки минус margin — предсказать его нельзя
    assert out[:, 1].tolist() == [-1.0 - ee.FLOOR_MARGIN, 0.5 - ee.FLOOR_MARGIN]
    assert (out.argmax(1) != 1).all()


def test_oof_logits_class_absent_from_train_part():
    X, y, folds = _toy()
    y = y.copy()
    only_fold0 = np.flatnonzero(y == "030.000.000")
    y[only_fold0] = "010.000.000"  # класс 030 остаётся единственным представителем в фолде 0:
    y[np.flatnonzero(folds == 0)[:3]] = "030.000.000"  # три строки, все в valid фолда 0
    scores, classes = ee.oof_logits(X, y, folds, C=1.0, max_iter=100)
    assert list(classes) == ["010.000.000", "020.000.000", "030.000.000"]
    v0 = folds == 0
    # в фолде 0 класса 030 в train-части нет: его оценка = пол, предсказать его нельзя
    assert (classes[scores[v0].argmax(1)] != "030.000.000").all()
    assert np.isfinite(scores).all()


def test_pick_blend_weight_sigma_rule():
    # (w, main_f1, control_f1): w=0 — только базовая модель
    rows = [(0.0, 0.8396, 0.8045), (0.3, 0.8440, 0.8090), (0.5, 0.8480, 0.8120)]
    # средний прирост w=0.5: ((0.8480-0.8396)+(0.8120-0.8045))/2 = 0.00795 < σ → нет прироста
    w, info = ee.pick_blend_weight(rows, sigma=0.0086)
    assert (
        w == 0.0 and info["gain"] == pytest.approx(0.00795, abs=1e-4) and info["improved"] is False
    )
    rows2 = [(0.0, 0.80, 0.78), (0.4, 0.83, 0.80), (0.6, 0.84, 0.81)]
    w2, info2 = ee.pick_blend_weight(rows2, sigma=0.0086)
    assert w2 == 0.6 and info2["improved"] is True  # лучший по сумме двух схем и прирост > σ


def test_load_aligned_embeddings(tmp_path):
    from itgorod.embeddings import EmbeddingSpec, cache_path

    spec = EmbeddingSpec("m/x", 512, "with", 5)
    np.save(cache_path(tmp_path, spec), np.zeros((5, 4), dtype=np.float16))
    X = ee.load_embeddings(tmp_path, spec)
    assert X.dtype == np.float32 and X.shape == (5, 4)
    with pytest.raises(FileNotFoundError):
        ee.load_embeddings(tmp_path, EmbeddingSpec("m/x", 512, "none", 5))


def test_run_embed_eval_synthetic(tmp_path):
    """Оркестратор на синтетике: подставные OOF-оценки TF-IDF и готовый кэш эмбеддингов."""
    import json

    from itgorod.embeddings import EmbeddingSpec

    n, k = 120, 3
    rng = np.random.default_rng(1)
    y = np.array(["010.000.000", "020.000.000", "030.000.000"])[np.arange(n) % k]
    orgs = pd.Series([f"орган {i % 12}" for i in range(n)], dtype=object)
    folds = {"main": np.arange(n) % 4, "control": (np.arange(n) % 12) % 4}
    classes = np.array(sorted(set(y)))
    onehot = (y[:, None] == classes[None, :]).astype(np.float32)
    fake_svc = {
        s: onehot * 2 + rng.normal(scale=0.5, size=(n, k)).astype(np.float32) for s in folds
    }
    for header in ("with", "none"):
        X = (onehot @ rng.normal(size=(k, 6)) * 3 + rng.normal(size=(n, 6))).astype(np.float16)
        np.save(_mk(tmp_path / "cache", EmbeddingSpec("m/x", 512, header, n)), X)
    res = ee.run_embed_eval(
        model="m/x",
        max_tokens=512,
        y=y,
        orgs=orgs,
        folds=folds,
        svc_scores=fake_svc,
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "out",
        log=lambda *_: None,
    )
    assert set(res["headers"]) == {"with", "none"}
    for r in res["headers"].values():
        assert set(r) >= {"lr", "chosen_w", "improved", "gain", "curve_csv"}
        assert (tmp_path / "out" / r["curve_csv"]).exists()
        assert r["lr"]["main_macro_f1"] > 0.5
    rows = pd.read_csv(tmp_path / "out" / "t2_embed_eval_m--x.csv")
    assert any(i.endswith("-lr") for i in rows["id"]) and any("-blend-" in i for i in rows["id"])
    assert json.loads((tmp_path / "out" / "t2_embed_eval_m--x.json").read_text())["sigma"] == 0.0086


def _mk(cache_dir, spec):
    from itgorod.embeddings import cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_path(cache_dir, spec)
