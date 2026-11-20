"""T2: настройка LinearSVC (усечение, голова+хвост, C, вклад шапки), основная схема для сетки,
контрольная — для подтверждения финалиста.

Правило выбора: кандидат лучше базы, только если выигрыш Macro-F1 больше σ по фолдам
(берётся большая из σ базы и кандидата); иначе улучшения нет. Выбор по той же CV, на которой
сравниваются конфиги, поэтому оценка победителя ОПТИМИСТИЧНА (выбор шумит на уровне σ).
"""

import dataclasses
import json
from pathlib import Path

import pandas as pd

from itgorod.config import T2Config
from itgorod.t2_experiment import (
    Experiment,
    header_parse_rate,
    load_meta,
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
