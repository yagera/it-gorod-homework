import pandas as pd
import pytest

from itgorod.t2_recon import (
    body_hash,
    class_stats,
    date_distribution,
    duplicate_report,
    head_hash,
    header_frame,
    hierarchy_summary,
    leading_code,
    leading_code_report,
    length_quantiles,
    org_overlap,
    parent_codes,
    parse_header,
    time_split_report,
)

DOC = "ПОСТАНОВЛЕНИЕ от 05.06.2025 № 125-па Администрация 27.01.2026 Текст документа тело"


def test_parent_codes():
    assert parent_codes("010.060.020") == ("010.000.000", "010.060.000", "010.060.020")


def test_hierarchy_parents_as_labels():
    h = hierarchy_summary(["010.000.000", "010.060.000", "010.060.020", "020.000.000"])
    assert (h["n_codes"], h["n_level1"], h["n_level2"]) == (4, 2, 3)
    assert h["parents_as_labels"] == ["010.000.000", "010.060.000"]
    assert h["leaf_only_codes"] == ["010.060.020", "020.000.000"]
    assert h["ancestors_missing_as_labels"] == []
    assert hierarchy_summary(["010.060.020"])["ancestors_missing_as_labels"] == [
        "010.000.000",
        "010.060.000",
    ]


def test_class_stats_rare():
    codes = pd.Series(["a"] * 100 + ["b"] * 10)
    table, s = class_stats(codes, rare_threshold=50)
    assert s["rare_classes"] == ["b"] and s["imbalance_max_min"] == 10
    assert table["share"].sum() == pytest.approx(1)


def test_head_hash_uses_head_only():
    assert head_hash("x" * 10 + "A", n_chars=10) == head_hash("x" * 10 + "B", n_chars=10)
    assert head_hash("a") != head_hash("b")


def test_duplicate_report():
    tr_h = pd.Series(["h1", "h1", "h2", "h3", "h3"])
    tr_c = pd.Series(["a", "b", "a", "c", "c"])
    te_h = pd.Series(["h1", "h9", "h9"])
    r = duplicate_report(tr_h, tr_c, te_h)
    assert r["train_rows_in_dup_groups"] == 4 and r["train_dup_groups"] == 2
    assert r["train_conflict_rows"] == 2 and r["train_conflict_groups"] == 1
    assert r["test_rows_in_train"] == 1 and r["test_dup_rows_inside"] == 2


def test_leading_code():
    assert leading_code(" 010.060.000 текст") == "010.060.000"
    assert leading_code("текст 010.060.000") is None
    r = leading_code_report(
        pd.Series(["010.060.000 a", "020.000.000 b", "нет 030.000.000"]),
        pd.Series(["010.060.000", "010.000.000", "030.000.000"]),
    )
    assert r["leading_code_rows"] == 2 and r["leading_equals_label_rows"] == 1
    assert r["anywhere_in_text_rows"] == 3


def test_parse_header():
    h = parse_header(DOC)
    assert h["type"] == "ПОСТАНОВЛЕНИЕ" and h["number"] == "125-па"
    assert h["act_date"] == pd.Timestamp("2025-06-05")
    assert h["fetch_date"] == pd.Timestamp("2026-01-27")


def test_parse_header_multiword_type_and_missing():
    h = parse_header("МУНИЦИПАЛЬНЫЙ ПРАВОВОЙ АКТ от 05.08.2025 № 166 Орган")
    assert h["type"] == "МУНИЦИПАЛЬНЫЙ ПРАВОВОЙ АКТ"
    bad = parse_header("без шапки")
    assert bad["type"] is None and pd.isna(bad["act_date"]) and pd.isna(bad["fetch_date"])
    assert pd.isna(parse_header("РЕШЕНИЕ от 31.02.2025 № 1")["act_date"])


def test_date_distribution_and_split():
    a = pd.Series(pd.to_datetime(["2025-01-10", "2025-02-01", None]))
    b = pd.Series(pd.to_datetime(["2025-01-20", "2026-03-01"]))
    d = date_distribution(a)
    assert d["parsed"] == 2 and d["by_month"] == {"2025-01": 1, "2025-02": 1}
    s = time_split_report(a, b)
    assert s["test_after_train_max"] == 0.5 and s["test_before_train_min"] == 0.0


def test_body_hash_ignores_header():
    a = "РЕШЕНИЕ от 01.01.2025 № 1 Орган А 20.01.2026 Текст документа одно и то же тело"
    b = "РЕШЕНИЕ от 02.02.2025 № 9 Орган Б 21.01.2026 Текст документа одно и то же тело"
    assert body_hash(a) == body_hash(b) and head_hash(a) != head_hash(b)
    assert body_hash("без маркера") == head_hash("без маркера")


def test_parse_header_org_and_frame():
    h = parse_header(DOC)
    assert h["org"] == "Администрация"
    df = header_frame(pd.Series([DOC, "мусор"], index=[5, 6]))
    assert list(df.index) == [5, 6] and pd.isna(df.loc[6, "org"])


def test_org_overlap():
    r = org_overlap(pd.Series(["Орган А", "Б"]), pd.Series(["орган а", "В", None]))
    assert r["test_rows_with_org"] == 2 and r["test_rows_org_in_train"] == 1


def test_code_re_boundaries_and_anywhere_label():
    r = leading_code_report(
        pd.Series(["номер 1.234.567.890 и 010.060.000 далее", "x 020.000.000"]),
        pd.Series(["010.060.000", "030.000.000"]),
    )
    assert r["anywhere_in_text_rows"] == 2 and r["anywhere_equals_label_rows"] == 1
    assert r["leading_code_rows"] == 0


def test_parent_codes_rejects_bad_format():
    with pytest.raises(ValueError):
        parent_codes("10.60.20")


def test_codes_keep_leading_zeros_as_str():
    table, _ = class_stats(pd.Series(["010.000.000", "010.000.000", "020.000.000"]))
    assert table["code"].tolist() == ["010.000.000", "020.000.000"]


def test_length_quantiles_and_empty_dates():
    q = length_quantiles(pd.Series([10, 20, 30]))
    assert q["min"] == 10 and q["max"] == 30 and q["q0.5"] == 20
    assert date_distribution(pd.Series([pd.NaT, pd.NaT]))["parsed"] == 0
    with pytest.raises(ValueError):
        time_split_report(pd.Series([pd.NaT]), pd.Series(pd.to_datetime(["2025-01-01"])))


def test_duplicate_report_test_unique():
    r = duplicate_report(pd.Series(["a"]), pd.Series(["x"]), pd.Series(["b", "b", "c"]))
    assert r["test_unique_hashes"] == 2
