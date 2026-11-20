import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from itgorod.t2_cv import (
    build_meta,
    control_folds,
    fit_fold_features,
    group_ids,
    main_folds,
    normalize_org,
    oof_report,
    seen_org_mask,
)


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.array(["010.000.000"] * (n // 2) + ["020.000.000"] * (n // 3))
    codes = np.concatenate([codes, np.array(["030.000.000"] * (n - len(codes)))])
    rng.shuffle(codes)
    orgs = np.array([f"org{i % 30}" for i in range(n)], dtype=object)
    bodies = np.array([f"h{i}" for i in range(n)], dtype=object)
    return codes, orgs, bodies


def test_normalize_org():
    assert (
        normalize_org("  Администрация «Берёзовского»  с.п. ") == "администрация берёзовского с п"
    )
    assert normalize_org("АДМИНИСТРАЦИЯ  берёзовского с.п") == "администрация берёзовского с п"
    assert normalize_org(None) is None and normalize_org("  «» ") is None


def test_build_meta():
    docs = [
        "РЕШЕНИЕ от 01.01.2025 № 1 Совет «А» 20.01.2026 Текст документа тело",
        "без шапки",
    ]
    m = build_meta(docs)
    assert m.loc[0, "org_norm"] == "совет а" and pd.isna(m.loc[1, "org_norm"])
    assert len(m.loc[0, "body_hash"]) == 16


def test_group_ids_union_and_missing_org():
    g = group_ids(["b1", "b2", "b3", "b4"], ["o1", "o1", None, None])
    assert g[0] == g[1]  # общий орган
    assert g[2] != g[3] and g[2] != g[0]  # без органа — свои группы
    g2 = group_ids(["b1", "b1", "b3"], ["o1", "o2", "o3"])  # общее тело связывает органы
    assert g2[0] == g2[1] != g2[2]


def test_main_folds_stratified_deterministic_dups_together():
    codes, orgs, bodies = _data()
    bodies[5] = bodies[77]  # точный дубль тела
    codes[77] = codes[5]
    f1 = main_folds(codes, bodies, 5, 42)
    f2 = main_folds(codes, bodies, 5, 42)
    assert (f1 == f2).all() and set(f1) == set(range(5))
    assert f1[5] == f1[77]
    for k in range(5):  # в каждом фолде есть каждый класс
        assert set(codes[f1 == k]) == set(codes)


def test_main_folds_conflicting_dups_still_one_fold():
    codes, orgs, bodies = _data()
    bodies[3] = bodies[90]
    codes[3], codes[90] = "010.000.000", "030.000.000"
    f = main_folds(codes, bodies, 5, 42)
    assert f[3] == f[90]


def test_control_folds_groups_not_split():
    codes, orgs, bodies = _data()
    bodies[10] = bodies[55]  # связывает два органа
    f = control_folds(codes, bodies, orgs, 5, 42)
    g = group_ids(bodies, orgs)
    for gid in set(g):
        assert len(set(f[g == gid])) == 1
    assert set(f) == set(range(5))
    # органы не пересекаются между train- и valid-частью
    for k in range(5):
        assert not (set(orgs[f == k]) & set(orgs[f != k]))


def test_seen_org_mask_uses_train_part_only():
    orgs = pd.Series(["a", "b", "c", None, "a"], dtype=object)
    folds = np.array([0, 0, 1, 1, 1])
    m = seen_org_mask(orgs, folds)
    # fold 0 valid: a (a есть в fold 1 train-части → seen), b (нет → unseen)
    # fold 1 valid: c (нет), None (unseen), a (есть в fold 0 → seen)
    assert m.tolist() == [True, False, False, False, True]


def test_oof_report_slices_and_rare():
    y = np.array(["a"] * 60 + ["b"] * 60 + ["r"] * 6)
    p = y.copy()
    p[:3] = "b"  # три ошибки в a
    seen = np.array([True] * 63 + [False] * 63)
    r = oof_report(y, p, seen, rare_threshold=50)
    assert r["rare_classes"] == ["r"] and r["macro_f1_rare"] == pytest.approx(1.0)
    assert 0 < r["macro_f1"] < 1 and r["n_seen"] == 63 and r["n_unseen"] == 63
    assert r["macro_f1_unseen"] == pytest.approx(1.0)  # ошибки только в seen
    assert set(r["per_class"]["code"]) == {"a", "b", "r"}


class SpyVec(TfidfVectorizer):
    fit_docs: list = []

    def fit_transform(self, raw_documents, y=None):
        SpyVec.fit_docs = list(raw_documents)
        return super().fit_transform(raw_documents, y)

    def fit(self, raw_documents, y=None):  # fit на чём-либо кроме train — нарушение
        SpyVec.fit_docs = list(raw_documents)
        return super().fit(raw_documents, y)


def test_fit_fold_features_fit_only_on_train():
    train = ["кошка собака", "собака дом", "кошка дом"]
    valid = ["уникальноеслово собака"]
    vec = SpyVec()
    xtr, xva = fit_fold_features(vec, train, valid)
    assert SpyVec.fit_docs == train
    assert "уникальноеслово" not in vec.vocabulary_
    assert xtr.shape[0] == 3 and xva.shape[0] == 1 and xtr.shape[1] == xva.shape[1]


# ---- группы по дате выгрузки (проверка стабильности) ----


def test_fetch_dates_extraction():
    from itgorod.t2_cv import fetch_dates

    texts = [
        "РЕШЕНИЕ от 01.02.2025 № 1 Орган 20.01.2026 Текст документа тело",
        "без шапки",
        "ПОСТАНОВЛЕНИЕ от 03.04.2025 № 2 Орган 02.02.2026 Текст документа тело",
    ]
    assert fetch_dates(texts) == ["20.01.2026", None, "02.02.2026"]


def test_date_group_folds_quantile_contiguous_and_whole_dates():
    from itgorod.t2_cv import date_group_folds

    # 8 дат по 10 строк: хронологические группы из целых дат, ~по 2 даты
    dates = [f"{d:02d}.01.2026" for d in range(19, 27)]
    fetch = [d for d in dates for _ in range(10)]
    bodies = [f"b{i}" for i in range(80)]
    f = date_group_folds(fetch, bodies, n_groups=4)
    assert sorted(set(f)) == [0, 1, 2, 3]
    for d in dates:  # дата целиком в одной группе
        assert len({f[i] for i, x in enumerate(fetch) if x == d}) == 1
    order = [f[fetch.index(d)] for d in dates]
    assert order == sorted(order)  # группы идут по времени
    assert all(15 <= (f == k).sum() <= 25 for k in range(4))  # квантили, ~25%


def test_date_group_folds_missing_dates_and_duplicates():
    from itgorod.t2_cv import date_group_folds

    dates = [f"{d:02d}.01.2026" for d in range(19, 23) for _ in range(10)] + [None, None]
    bodies = [f"b{i}" for i in range(42)]
    bodies[41] = bodies[0]  # дубль тела: строка без даты должна уйти в фолд первой копии
    f = date_group_folds(dates, bodies, n_groups=4)
    assert len(f) == 42 and set(f) <= {0, 1, 2, 3}
    assert f[40] == 3  # без даты → последняя группа
    assert f[41] == f[0]  # дубль тела уходит в фолд первой копии, даже без даты


# ---- подмена даты выгрузки в valid-текстах ----


def test_swap_fetch_date_only_the_header_date():
    from itgorod.t2_cv import swap_fetch_date

    t = (
        "РЕШЕНИЕ от 20.01.2026 № 5 Орган А 20.01.2026 Текст документа "
        "тело 20.01.2026 и снова 20.01.2026"
    )
    out = swap_fetch_date(t, "31.01.2026")
    assert out == (
        "РЕШЕНИЕ от 20.01.2026 № 5 Орган А 31.01.2026 Текст документа "
        "тело 20.01.2026 и снова 20.01.2026"
    )
    assert swap_fetch_date("без шапки", "31.01.2026") == "без шапки"  # даты нет — без изменений


def test_swap_valid_dates_pool_seed_and_count():
    from itgorod.t2_cv import swap_valid_dates

    valid = [
        f"РЕШЕНИЕ от 01.01.2025 № {i} Орган 19.01.2026 Текст документа тело" for i in range(50)
    ]
    valid.append("без шапки совсем")
    pool = ["25.01.2026", "26.01.2026", "27.01.2026"]
    a, n_a = swap_valid_dates(valid, pool, seed=42)
    b, n_b = swap_valid_dates(valid, pool, seed=42)
    c, _ = swap_valid_dates(valid, pool, seed=7)
    assert a == b and n_a == n_b and a != c  # детерминированно по seed
    assert a[-1] == valid[-1]  # документ без даты не тронут
    assert n_a == 50  # у всех 50 документов дата заменена (пул не содержит исходную)
    for t in a[:-1]:
        assert any(f" {d} Текст документа" in t for d in pool)
    assert len(a) == len(valid)
