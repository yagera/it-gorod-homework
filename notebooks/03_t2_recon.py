# %% [markdown]
# # T2: разведка (схема, классы, шапки, даты, дубли, код в тексте)
# Запуск: `uv run python notebooks/03_t2_recon.py [--limit N]`. Тексты читаются батчами и
# усекаются на стороне Arrow (`iter_t2_batches`); в памяти держатся только усечённые тексты
# батча, длины, хеши и поля шапки. Итог — JSON в `data/processed/t2_recon.json` и печать.

# %%
import argparse
import json
import resource
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from itgorod.config import REPO_ROOT, load_t2_config
from itgorod.data import iter_t2_batches
from itgorod.t2_recon import (
    body_hash,
    class_stats,
    date_distribution,
    duplicate_report,
    head_hash,
    hierarchy_summary,
    leading_code_report,
    length_quantiles,
    org_overlap,
    parse_header,
    time_split_report,
)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None, help="куда писать JSON (по умолчанию data/processed)")
ap.add_argument("--limit", type=int, default=None, help="первые N строк каждого файла (отладка)")
args = ap.parse_args()

cfg = load_t2_config(REPO_ROOT / "configs" / "t2.yaml")
HASH_CHARS = 2000
t0 = time.time()


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: КБ → МБ


def scan(path, columns):
    """Потоково: длины, хеши головы, шапка, доля кода в начале; без хранения текстов."""
    parts, leading = [], {"n": 0, "leading_code_rows": 0, "anywhere_in_text_rows": 0}
    lead_eq = lead_diff = any_eq = 0
    head_samples = []
    for df in iter_t2_batches(path, columns, cfg.batch_size, cfg.max_chars, args.limit):
        hdr = pd.DataFrame([parse_header(t) for t in df["text"]])
        part = pd.DataFrame(
            {
                "len": df["text"].str.len().to_numpy(),
                "hash": [head_hash(t, HASH_CHARS) for t in df["text"]],
                "body_hash": [body_hash(t, HASH_CHARS) for t in df["text"]],
            }
        )
        part = pd.concat([part, hdr], axis=1)
        if "code" in df:
            part["code"] = df["code"].to_numpy()
        if "row_id" in df:
            part["row_id"] = df["row_id"].to_numpy()
        parts.append(part)
        r = leading_code_report(df["text"], df["code"] if "code" in df else None)
        for k in ("n", "leading_code_rows", "anywhere_in_text_rows"):
            leading[k] += r[k]
        any_eq += r.get("anywhere_equals_label_rows", 0)
        lead_eq += r.get("leading_equals_label_rows", 0)
        lead_diff += r.get("leading_differs_from_label_rows", 0)
        if len(head_samples) < 5:
            head_samples.append(df["text"].iloc[0][:300])
    leading["anywhere_equals_label_rows"] = any_eq
    leading["leading_equals_label_rows"] = lead_eq
    leading["leading_differs_from_label_rows"] = lead_diff
    return pd.concat(parts, ignore_index=True), leading, head_samples


# %% 1. Схема
rep = {"limit": args.limit, "hash_chars": HASH_CHARS, "max_chars": cfg.max_chars}
for name, p in (("train", cfg.paths.train), ("test", cfg.paths.test)):
    pf = pq.ParquetFile(p)
    rep[f"schema_{name}"] = {
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "columns": {f.name: str(f.type) for f in pf.schema_arrow},
    }
sample = pd.read_csv(cfg.paths.sample, dtype=str)
rep["sample"] = {
    "rows": len(sample),
    "columns": list(sample.columns),
    "unique_codes": sorted(sample["code"].dropna().unique().tolist()),
}

# %% 2–6. Проходы по данным
train, lead_train, heads_train = scan(cfg.paths.train, ["text", "code"])
rep["peak_mb_after_train"] = peak_mb()
test, lead_test, heads_test = scan(cfg.paths.test, ["row_id", "text"])
rep["peak_mb"] = peak_mb()
rep["elapsed_s"] = round(time.time() - t0, 1)

# row_id
rep["row_id"] = {
    "dtype_test": str(test["row_id"].dtype),
    "unique": bool(test["row_id"].is_unique),
    "min": int(test["row_id"].min()),
    "max": int(test["row_id"].max()),
    "same_order_as_sample": bool(
        (test["row_id"].astype(str).to_numpy() == sample["row_id"].to_numpy()).all()
    )
    if len(test) == len(sample)
    else None,
}

# классы
table, cs = class_stats(train["code"], rare_threshold=50)
rep["classes"] = cs | {
    "top10": table.head(10).to_dict("records"),
    "bottom10": table.tail(10).to_dict("records"),
}
rep["hierarchy"] = hierarchy_summary(train["code"])
_n = table.set_index("code")["n"]
rep["hierarchy"]["parents_counts"] = {c: int(_n[c]) for c in rep["hierarchy"]["parents_as_labels"]}
rep["classes"]["rare_counts"] = {c: int(_n[c]) for c in rep["classes"]["rare_classes"]}
rep["codes_not_str"] = int((~train["code"].map(lambda c: isinstance(c, str))).sum())

# тексты
rep["len_train"] = length_quantiles(train["len"]) | {
    "share_at_cap": float((train["len"] >= cfg.max_chars).mean())
}
rep["len_test"] = length_quantiles(test["len"]) | {
    "share_at_cap": float((test["len"] >= cfg.max_chars).mean())
}
rep["header_samples_train"] = heads_train
rep["header_types_train"] = train["type"].value_counts(dropna=False).head(10).to_dict()
rep["header_types_test"] = test["type"].value_counts(dropna=False).head(10).to_dict()

# даты
for col in ("act_date", "fetch_date"):
    rep[f"{col}_train"] = date_distribution(train[col])
    rep[f"{col}_test"] = date_distribution(test[col])
    rep[f"{col}_split"] = time_split_report(train[col], test[col])
# связь даты с классом: доля топ-класса по месяцам (месяцы с ≥ 200 документами)
_m = train.dropna(subset=["act_date"]).assign(
    m=lambda d: d["act_date"].dt.to_period("M").astype(str)
)
_big = _m.groupby("m")["code"].transform("size") >= 200
_shares = _m[_big].groupby("m")["code"].value_counts(normalize=True).unstack(fill_value=0)
rep["act_month_vs_class"] = {
    "months_used": int(_shares.shape[0]),
    "top_class_share_by_month": {m: float(v) for m, v in _shares["010.000.000"].round(3).items()},
    "max_abs_share_diff_vs_overall_pp": float(
        ((_shares - train["code"].value_counts(normalize=True)).abs().max().max()) * 100
    ),
}

# дубли
rep["duplicates"] = duplicate_report(train["hash"], train["code"], test["hash"])
grp = train.groupby("hash")["code"].nunique()
rep["duplicates_body"] = duplicate_report(train["body_hash"], train["code"], test["body_hash"])
rep["orgs"] = org_overlap(train["org"], test["org"])
rep["orgs"]["train_missing_org"] = int(train["org"].isna().sum())
rep["orgs"]["test_missing_org"] = int(test["org"].isna().sum())

# код в тексте
rep["leading_code_train"] = lead_train
rep["leading_code_test"] = lead_test

default = "t2_recon.json" if args.limit is None else "t2_recon_limited.json"
out = Path(args.out) if args.out else cfg.paths.processed_dir / default
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
print(f"\nсохранено: {out}; пик RSS {rep['peak_mb']:.0f} МБ; {rep['elapsed_s']} с")
