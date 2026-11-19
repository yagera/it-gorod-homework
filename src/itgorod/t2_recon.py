"""T2: функции разведки (классы, иерархия кодов, дубли по хешу головы, код в тексте).

Всё работает на усечённых текстах; полный `text` не нужен и не загружается.
"""

import hashlib
import re
from collections.abc import Iterable

import pandas as pd

CODE_RE = re.compile(r"(?<![\d.])\d{3}\.\d{3}\.\d{3}(?![\d.])")
LEADING_CODE_RE = re.compile(r"^\s*(\d{3}\.\d{3}\.\d{3})")


def parent_codes(code: str) -> tuple[str, str, str]:
    """`NNN.MMM.KKK` → (уровень 1 `NNN.000.000`, уровень 2 `NNN.MMM.000`, сам код)."""
    if not CODE_RE.fullmatch(code):
        raise ValueError(f"код не в формате NNN.NNN.NNN: {code!r}")
    a, b, _ = code.split(".")
    return f"{a}.000.000", f"{a}.{b}.000", code


def hierarchy_summary(codes: Iterable[str]) -> dict:
    """Уникальные метки по уровням и «родители-метки»: коды, которые сами являются меткой и
    при этом префикс другой метки (например `010.060.000` при наличии `010.060.020`).
    Уровень 2 включает коды вида `NNN.000.000`, поэтому n_level1 + n_level2 ≠ n_codes."""
    uniq = sorted(set(codes))
    l1 = {parent_codes(c)[0] for c in uniq}
    l2 = {parent_codes(c)[1] for c in uniq}
    labels = set(uniq)
    ancestors = {a for c in uniq for a in parent_codes(c)[:2] if a != c}
    parents = sorted(ancestors & labels)
    return {
        "n_codes": len(uniq),
        "n_level1": len(l1),
        "n_level2": len(l2),
        "parents_as_labels": parents,
        "leaf_only_codes": [c for c in uniq if c not in parents],
        "ancestors_missing_as_labels": sorted(ancestors - labels),
    }


def org_overlap(train_orgs: pd.Series, test_orgs: pd.Series) -> dict:
    """Пересечение органов-издателей (строка между номером акта и датой выгрузки)."""
    tr = set(train_orgs.dropna().str.casefold())
    te = test_orgs.dropna().str.casefold()
    return {
        "train_unique_orgs": len(tr),
        "test_unique_orgs": int(te.nunique()),
        "test_rows_with_org": len(te),
        "test_rows_org_in_train": int(te.isin(tr).sum()),
        "test_rows_org_in_train_share": float(te.isin(tr).mean()) if len(te) else 0.0,
    }


def class_stats(codes: pd.Series, rare_threshold: int = 50) -> tuple[pd.DataFrame, dict]:
    """Таблица (code, n, share) по убыванию n и сводка дисбаланса."""
    vc = codes.value_counts()
    table = pd.DataFrame({"code": vc.index.astype(str), "n": vc.to_numpy()})
    table["share"] = table["n"] / table["n"].sum()
    summary = {
        "n_classes": len(table),
        "n_rows": int(table["n"].sum()),
        "max_n": int(table["n"].max()),
        "min_n": int(table["n"].min()),
        "median_n": float(table["n"].median()),
        "imbalance_max_min": float(table["n"].max() / table["n"].min()),
        "rare_threshold": rare_threshold,
        "rare_classes": table.loc[table["n"] < rare_threshold, "code"].tolist(),
    }
    return table, summary


BODY_MARKER = "Текст документа"


def body_hash(text: str, n_chars: int = 2000) -> str:
    """Хеш первых `n_chars` символов тела — после маркера «Текст документа», то есть без шапки
    (номер, дата, орган, дата выгрузки). Без маркера хешируется весь текст."""
    _, sep, body = text.partition(BODY_MARKER)
    return head_hash(body if sep else text, n_chars)


def head_hash(text: str, n_chars: int = 2000) -> str:
    """Компактный хеш первых `n_chars` символов (8 байт, hex) — для потокового поиска дублей."""
    return hashlib.blake2b(text[:n_chars].encode("utf-8"), digest_size=8).hexdigest()


def duplicate_report(
    train_hashes: pd.Series, train_codes: pd.Series, test_hashes: pd.Series
) -> dict:
    """Дубли по хешу головы: внутри train, train∩test и конфликты меток среди дублей train."""
    df = pd.DataFrame({"h": train_hashes.to_numpy(), "code": train_codes.to_numpy()})
    sizes = df.groupby("h")["code"].transform("size")
    n_codes = df.groupby("h")["code"].transform("nunique")
    dup_mask = sizes > 1
    conflict_mask = n_codes > 1
    train_set = set(df["h"])
    test_in_train = test_hashes.isin(train_set)
    return {
        "train_rows": len(df),
        "train_unique_hashes": int(df["h"].nunique()),
        "train_rows_in_dup_groups": int(dup_mask.sum()),
        "train_dup_groups": int(df.loc[dup_mask, "h"].nunique()),
        "train_max_group_size": int(sizes.max()),
        "train_conflict_rows": int(conflict_mask.sum()),
        "train_conflict_groups": int(df.loc[conflict_mask, "h"].nunique()),
        "test_rows": len(test_hashes),
        "test_rows_in_train": int(test_in_train.sum()),
        "test_unique_hashes": int(test_hashes.nunique()),
        "test_dup_rows_inside": int(test_hashes.duplicated(keep=False).sum()),
    }


def leading_code(text: str) -> str | None:
    """Код вида NNN.NNN.NNN в самом начале текста (после пробелов) или None."""
    m = LEADING_CODE_RE.match(text)
    return m.group(1) if m else None


def leading_code_report(texts: pd.Series, codes: pd.Series | None = None) -> dict:
    """Как часто текст начинается с кода; для train — как часто он совпадает с меткой.
    Также считает вхождения кода в первых символах текста в любом месте."""
    found = texts.map(leading_code)
    has = found.notna()
    out = {"n": len(texts), "leading_code_rows": int(has.sum())}
    out["anywhere_in_text_rows"] = int(texts.str.contains(CODE_RE).sum())
    if codes is not None:
        anywhere_eq = [c in set(CODE_RE.findall(t)) for t, c in zip(texts, codes, strict=True)]
        out["anywhere_equals_label_rows"] = int(sum(anywhere_eq))
        match = has & (found == codes)
        out["leading_equals_label_rows"] = int(match.sum())
        out["leading_differs_from_label_rows"] = int((has & ~match).sum())
    return out


def length_quantiles(lengths: pd.Series, qs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    return {f"q{q}": float(lengths.quantile(q)) for q in qs} | {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }


HEADER_RE = re.compile(
    r"^\s*(?P<type>[А-ЯЁ][А-ЯЁ ]*?)\s+от\s+(?P<act>\d{2}\.\d{2}\.\d{4})\s+№\s*(?P<num>\S+)"
)
FETCH_RE = re.compile(r"(?P<fetch>\d{2}\.\d{2}\.\d{4})\s+Текст документа")


def _to_date(s: str) -> pd.Timestamp:
    """DD.MM.YYYY → Timestamp; невозможная дата (31.02) → NaT."""
    return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")


def parse_header(text: str) -> dict:
    """Шапка документа: тип акта, дата акта и номер (`ТИП от DD.MM.YYYY № N`), дата выгрузки
    (`DD.MM.YYYY Текст документа`). Не найдено → None / NaT."""
    m = HEADER_RE.match(text)
    f = FETCH_RE.search(text[:1000])
    org = text[m.end() : f.start()].strip() if m and f and f.start() >= m.end() else None
    return {
        "org": org or None,
        "type": m.group("type").strip() if m else None,
        "act_date": _to_date(m.group("act")) if m else pd.NaT,
        "number": m.group("num") if m else None,
        "fetch_date": _to_date(f.group("fetch")) if f else pd.NaT,
    }


def header_frame(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([parse_header(t) for t in texts], index=texts.index)


def date_distribution(dates: pd.Series) -> dict:
    """Охват и разбивка по месяцам; доля непарсящихся."""
    ok = dates.dropna()
    if ok.empty:
        return {
            "n": len(dates),
            "parsed": 0,
            "parsed_share": 0.0,
            "min": None,
            "max": None,
            "by_month": {},
        }
    return {
        "n": len(dates),
        "parsed": len(ok),
        "parsed_share": len(ok) / len(dates) if len(dates) else 0.0,
        "min": str(ok.min().date()) if len(ok) else None,
        "max": str(ok.max().date()) if len(ok) else None,
        "by_month": ok.dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
    }


def time_split_report(train_dates: pd.Series, test_dates: pd.Series) -> dict:
    """Есть ли временной сплит: перекрытие диапазонов и доля test до/после границ train."""
    tr, te = train_dates.dropna(), test_dates.dropna()
    if tr.empty or te.empty:
        raise ValueError("нет распарсенных дат в train или test")
    return {
        "train_min": str(tr.min().date()),
        "train_max": str(tr.max().date()),
        "test_min": str(te.min().date()),
        "test_max": str(te.max().date()),
        "test_before_train_min": float((te < tr.min()).mean()),
        "test_after_train_max": float((te > tr.max()).mean()),
        "train_q05_q50_q95": [str(tr.quantile(q).date()) for q in (0.05, 0.5, 0.95)],
        "test_q05_q50_q95": [str(te.quantile(q).date()) for q in (0.05, 0.5, 0.95)],
    }
