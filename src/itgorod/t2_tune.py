"""T2: настройка LinearSVC (усечение, голова+хвост, C, вклад шапки), основная схема для сетки,
контрольная — для подтверждения финалиста.

Правило выбора: кандидат лучше базы, только если выигрыш Macro-F1 больше σ по фолдам
(берётся большая из σ базы и кандидата); иначе улучшения нет. Выбор по той же CV, на которой
сравниваются конфиги, поэтому оценка победителя ОПТИМИСТИЧНА (выбор шумит на уровне σ).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from itgorod.config import T2Config
from itgorod.t2_cv import date_group_folds, fetch_dates
from itgorod.t2_experiment import (
    Experiment,
    header_parse_rate,
    load_meta,
    load_train,
    run_experiments,
)

BASE = Experiment("svc", "none", 2000)
CS = (0.25, 1.0, 2.0)  # C=0.5 — это сама точка усечения (уже посчитана)
MAIN = "main_macro_f1"
STD = "main_fold_std"


def truncation_grid() -> list[Experiment]:
    return [
        Experiment("svc", "none", 1000),
        Experiment("svc", "none", 3000),
        Experiment("svc", "none", 1500, tail_chars=500),
        Experiment("svc", "none", 2000, tail_chars=1000),
    ]


def c_grid(best: Experiment, cs=CS) -> list[Experiment]:
    return [dataclasses.replace(best, C=c) for c in cs]


def row_to_experiment(row) -> Experiment:
    def get(key, default):  # строки старого CSV (до настройки) не содержат новых колонок
        v = row.get(key, default)
        return default if pd.isna(v) else v

    return Experiment(
        row["model"],
        row["class_weight"],
        int(row["max_chars"]),
        int(get("tail_chars", 0)),
        bool(get("strip_header", False)),
        float(get("C", 0.5)),
        str(get("header_mode", "")),
    )


def pick_best(base: dict, cands: list[dict], f1: str = MAIN, std: str = STD) -> dict:
    """База или лучший из кандидатов, обошедших базу больше чем на σ (max из двух σ)."""
    ok = [c for c in cands if c[f1] - base[f1] > max(base[std], c[std])]
    return max(ok, key=lambda c: c[f1]) if ok else base


def confirmed_by_control(base: dict, finalist: dict) -> bool:
    """Подтверждает ли контрольная схема выбор: прирост control Macro-F1 больше σ."""
    if finalist["id"] == base["id"]:
        return True
    gain = finalist["control_macro_f1"] - base["control_macro_f1"]
    return bool(gain > max(base["control_fold_std"], finalist["control_fold_std"]))


def pick_smallest_within_sigma(rows: list[dict], sigma: float, f1: str = MAIN) -> dict:
    """Наименьший C среди тех, чей Macro-F1 отличается от лучшего не больше чем на `sigma`
    (граница включительно)."""
    best = max(r[f1] for r in rows)
    ok = [r for r in rows if best - r[f1] <= sigma + 1e-12]
    return min(ok, key=lambda r: r["C"])


def big_drops(pc: pd.DataFrame, ref_id: str, var_id: str, scheme: str, thr: float) -> dict:
    """Классы, у которых F1 варианта ниже эталона больше чем на `thr` (дельта, отрицательная)."""

    def f1(i):
        x = pc[(pc["id"] == i) & (pc["scheme"] == scheme)]
        return x.set_index("code")["f1"]

    delta = (f1(var_id) - f1(ref_id)).dropna()
    return {c: round(float(d), 4) for c, d in delta.items() if d < -thr}


def ablation_grid(chosen: Experiment) -> list[Experiment]:
    """Вариант без шапки и 7 вариантов частей шапки (тело без шапки + выбранные части)."""
    base = dataclasses.replace(chosen, strip_header=True, tail_chars=0)
    modes = [
        "",
        "full",
        "only_type",
        "only_org",
        "only_numdates",
        "no_type",
        "no_org",
        "no_numdates",
    ]
    return [dataclasses.replace(base, header_mode=m) for m in modes]


def _rows(out_csv: Path) -> dict[str, dict]:
    df = pd.read_csv(out_csv, dtype={"strip_header": bool})
    return {r["id"]: r for r in df.to_dict("records")}


def _runner(cfg: T2Config, out_csv: Path, per_class_csv: Path, log):
    """`run(exps, schemes)`: считает только то, чего нет в CSV для нужных схем; строки с другим
    `n_rows` (прогон с --limit) не принимаются. Прогон детерминирован — это позволяет
    продолжать после остановки; версию кода и параметры признаков строка не фиксирует."""

    def run(exps, schemes=("main",)):
        have = _rows(out_csv) if out_csv.exists() else {}
        n_rows = len(load_meta(cfg))
        need = [
            e
            for e in exps
            if e.id not in have
            or have[e.id].get("n_rows") != n_rows
            or any(pd.isna(have[e.id].get(f"{s}_macro_f1")) for s in schemes)
        ]
        if len(need) < len(exps):
            log(f"пропуск (уже в CSV): {[e.id for e in exps if e not in need]}")
        if need:
            run_experiments(
                cfg, need, out_csv, per_class_csv, log=log, scheme_names=schemes, append=True
            )
        return _rows(out_csv)

    return run


def run_tune(
    cfg: T2Config,
    out_csv: Path,
    per_class_csv: Path,
    summary_json: Path,
    log=print,
    base: Experiment = BASE,
    truncations: list[Experiment] | None = None,
    cs=CS,
) -> dict:
    truncations = truncation_grid() if truncations is None else truncations

    run = _runner(cfg, out_csv, per_class_csv, log)

    # этап 0: база с обеими схемами (если ещё нет)
    rows = _rows(out_csv) if out_csv.exists() else {}
    if base.id not in rows or pd.isna(rows[base.id].get("control_macro_f1")):
        rows = run([base], ("main", "control"))
    base_row = rows[base.id]
    log(f"база {base.id}: main {base_row[MAIN]} σ {base_row[STD]}")

    # этап 1: усечение и «голова + хвост», C = 0.5
    rows = run(truncations)
    cands = [rows[e.id] for e in truncations]
    best_trunc = pick_best(base_row, cands)
    log(f"усечение: выбран {best_trunc['id']}")

    # этап 2: C на лучшем усечении (C=0.5 — сама строка best_trunc)
    best_exp = row_to_experiment(best_trunc)
    c_exps = c_grid(best_exp, cs)
    rows = run(c_exps)
    best_c = pick_best(best_trunc, [rows[e.id] for e in c_exps])
    log(f"C: выбран {best_c['id']}")

    # этап 3: вклад шапки на лучшем конфиге (информативно, в финалиста автоматически не идёт)
    finalist_exp = row_to_experiment(best_c)
    nohdr = dataclasses.replace(finalist_exp, strip_header=True)
    rows = run([nohdr])
    header_row = rows[nohdr.id]

    # этап 4: контрольная схема для финалиста
    if finalist_exp.id != base.id:
        rows = run([finalist_exp], ("main", "control"))
    fin = rows[finalist_exp.id]
    base_row = rows[base.id]
    confirmed = confirmed_by_control(base_row, fin)

    summary = {
        "base": {"id": base.id, "main": base_row[MAIN], "std": base_row[STD]},
        "truncation": {"chosen": best_trunc["id"], "candidates": {c["id"]: c[MAIN] for c in cands}},
        "c": {
            "chosen": best_c["id"],
            "candidates": {i: r[MAIN] for i, r in rows.items() if i in {e.id for e in c_exps}},
        },
        "header": {
            "id": nohdr.id,
            "main": header_row[MAIN],
            "std": header_row[STD],
            "delta_vs_finalist": round(header_row[MAIN] - fin[MAIN], 4),
        },
        "finalist": fin["id"],
        "finalist_main": fin[MAIN],
        "finalist_control": fin["control_macro_f1"],
        "base_control": base_row["control_macro_f1"],
        "confirmed": confirmed,
        "note": "выбор по той же CV, оценка победителя оптимистична",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    return summary


C_EXTRA = (4.0, 8.0)
SIGMA_FIXED = 0.0086  # σ базы по основной схеме (записи 14, 16)


def run_tune_c_and_header(
    cfg: T2Config,
    out_csv: Path,
    per_class_csv: Path,
    summary_json: Path,
    log=print,
    base: Experiment = BASE,
    extra_cs=C_EXTRA,
    sigma: float = SIGMA_FIXED,
    drop_thr: float = 0.10,
) -> dict:
    """Этап 2 настройки: C ∈ {4, 8} (основная схема), выбор наименьшего C в пределах σ от
    лучшего, подтверждение контрольной; абляция шапки по частям на выбранном C в обеих
    схемах. Выбор по той же CV — оценка оптимистична."""

    run = _runner(cfg, out_csv, per_class_csv, log)

    # (1) C: доп. значения на основной схеме; все C этого варианта текста — кандидаты
    extra = c_grid(base, extra_cs)
    rows = run([base, *c_grid(base, (0.25, 1.0, 2.0)), *extra])
    c_exps = [base, *c_grid(base, (0.25, 1.0, 2.0)), *extra]
    c_rows = [rows[e.id] for e in c_exps if e.id in rows]
    chosen_row = pick_smallest_within_sigma(c_rows, sigma)
    chosen = row_to_experiment(chosen_row)
    log(f"C: выбран {chosen.id} (лучший {max(r[MAIN] for r in c_rows)}, σ {sigma})")

    # (2) подтверждение контрольной: выбранный C против базы C=0.5 и против прошлого финалиста
    rows = run([chosen, base], ("main", "control"))
    fin, base_row = rows[chosen.id], rows[base.id]
    confirmed = confirmed_by_control(base_row, fin)
    prev = rows.get(dataclasses.replace(base, C=2.0).id)
    prev_control = prev.get("control_macro_f1") if prev is not None else None

    # (3) абляция шапки на выбранном C, обе схемы
    grid = ablation_grid(chosen)
    rows = run(grid, ("main", "control"))
    pc = pd.read_csv(per_class_csv, dtype={"code": str})
    ablation = {}
    for e in grid:
        r = rows[e.id]
        entry = {
            "id": e.id,
            "main": r[MAIN],
            "control": r["control_macro_f1"],
            "delta_main_vs_with_header": round(r[MAIN] - fin[MAIN], 4),
            "delta_control_vs_with_header": round(
                r["control_macro_f1"] - fin["control_macro_f1"], 4
            ),
        }
        for scheme in ("main", "control"):
            entry[f"drops_gt_{int(drop_thr * 100)}pp_{scheme}"] = big_drops(
                pc, chosen.id, e.id, scheme, drop_thr
            )
        ablation[e.header_mode or "none"] = entry

    summary = {
        "chosen": chosen.id,
        "chosen_C": chosen.C,
        "c_candidates": {r["id"]: r[MAIN] for r in c_rows},
        "sigma": sigma,
        "rule": "наименьший C в пределах σ от лучшего (main)",
        "finalist_main": fin[MAIN],
        "finalist_control": fin["control_macro_f1"],
        "base_control": base_row["control_macro_f1"],
        "prev_finalist_C2_control": prev_control,
        "confirmed": confirmed,
        "ablation": ablation,
        "parse_rate": header_parse_rate(cfg),
        "note": "выбор C по той же CV, оценка оптимистична; абляция — разрез тех же данных",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    return summary


def class_group_spread(y, folds) -> dict:
    """Насколько классы разнесены по группам: классы в одной группе, в нескольких, и доля
    строк, чей класс отсутствует в train-части своего фолда."""
    y, folds = np.asarray(y), np.asarray(folds)
    groups = pd.Series(folds).groupby(y).nunique()
    multi = sorted(groups[groups > 1].index)
    absent_rows = 0
    for k in np.unique(folds):
        absent_rows += int((~np.isin(y[folds == k], y[folds != k])).sum())
    return {
        "n_single_group": int((groups == 1).sum()),
        "n_multi_group": len(multi),
        "multi_group_classes": multi,
        "rows_share_class_absent_from_train_part": round(absent_rows / len(y), 4),
    }


def mean_f1_on_classes(pc: pd.DataFrame, id_: str, scheme: str, classes: list[str]) -> float:
    """Средний F1 (по OOF) выбранных классов для эксперимента и схемы."""
    x = pc[(pc["id"] == id_) & (pc["scheme"] == scheme) & pc["code"].isin(classes)]
    return float(x["f1"].mean())


def stability_grid(c: float = 1.0) -> list[Experiment]:
    """Финалист (с шапкой), «только номер и даты» и «без номера и дат» (тело без шапки + часть)."""
    return [
        Experiment("svc", "none", 2000, C=c),
        Experiment("svc", "none", 2000, strip_header=True, C=c, header_mode="only_numdates"),
        Experiment("svc", "none", 2000, strip_header=True, C=c, header_mode="no_numdates"),
    ]


def run_date_stability(
    cfg: T2Config,
    out_csv: Path,
    per_class_csv: Path,
    summary_json: Path,
    log=print,
    exps: list[Experiment] | None = None,
    fetch=None,
    n_groups: int = 4,
) -> dict:
    """Проверка стабильности: группы по дате выгрузки (квантили), обучение на одних группах,
    оценка на другой (group k-fold). Только оценка — сабмишн не пишется. Сравнивается с
    основной (main) и контрольной (control) схемами из CSV."""
    exps = stability_grid() if exps is None else exps
    meta = load_meta(cfg)
    if fetch is None:
        fetch = fetch_dates(load_train(cfg, 2000)["text"].tolist())
    if len(fetch) != len(meta):
        raise ValueError(f"даты выгрузки ({len(fetch)}) и мета ({len(meta)}) не совпадают по длине")
    folds = date_group_folds(fetch, meta["body_hash"].tolist(), n_groups)
    y = meta["code"].to_numpy()
    sizes = np.bincount(folds, minlength=n_groups).tolist()
    log(f"группы по дате выгрузки: размеры {sizes}")

    have = _rows(out_csv) if out_csv.exists() else {}
    need = [
        e
        for e in exps
        if e.id not in have
        or have[e.id].get("n_rows") != len(meta)
        or pd.isna(have[e.id].get("date_macro_f1"))
    ]
    if need:
        run_experiments(
            cfg,
            need,
            out_csv,
            per_class_csv,
            log=log,
            scheme_names=("date",),
            append=True,
            extra_schemes={"date": folds},
        )
    rows = _rows(out_csv)
    absent = {}
    for k in range(n_groups):
        tr_classes = set(y[folds != k])
        miss = sorted(set(y[folds == k]) - tr_classes)
        absent[k] = {"classes": miss, "rows": int(np.isin(y[folds == k], miss).sum())}
    variants = {}
    for e in exps:
        r = rows[e.id]
        variants[e.id] = {
            "date_macro_f1": r["date_macro_f1"],
            "date_fold_std": r["date_fold_std"],
            "date_f1_rare": r["date_f1_rare"],
            "main_macro_f1": r.get("main_macro_f1"),
            "control_macro_f1": r.get("control_macro_f1"),
            "main_f1_rare": r.get("main_f1_rare"),
            "fold_f1": json.loads(r["date_fold_f1"]),
            "absent_classes_by_fold": absent,
        }
    spread = class_group_spread(y, folds)
    pcdf = pd.read_csv(per_class_csv, dtype={"code": str})
    multi = spread["multi_group_classes"]
    for e in exps:
        v = variants[e.id]
        v["mean_f1_multi_group_classes_date"] = round(
            mean_f1_on_classes(pcdf, e.id, "date", multi), 4
        )
        has_main = ((pcdf["id"] == e.id) & (pcdf["scheme"] == "main")).any()
        v["mean_f1_multi_group_classes_main"] = (
            round(mean_f1_on_classes(pcdf, e.id, "main", multi), 4) if has_main else None
        )
    summary = {
        "class_spread": spread,
        "groups": {
            "n_groups": n_groups,
            "sizes": sizes,
            "rule": "хронологические квантильные группы из целых дат",
        },
        "variants": variants,
        "note": "только оценка; сабмишн не менялся",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    return summary
