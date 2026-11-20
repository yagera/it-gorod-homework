import json

import pandas as pd

from itgorod import t2_tune as tt
from itgorod.t2_experiment import Experiment


def _row(id_, f1, std, **kw):
    base = {"id": id_, "main_macro_f1": f1, "main_fold_std": std}
    return base | kw


def test_pick_best_requires_gain_over_sigma():
    base = _row("base", 0.829, 0.009)
    # выигрыш 0.008 < σ 0.009 → улучшения нет
    assert tt.pick_best(base, [_row("a", 0.837, 0.006)])["id"] == "base"
    # выигрыш 0.012 > max(σ) = 0.009 → принимаем
    assert tt.pick_best(base, [_row("a", 0.841, 0.006)])["id"] == "a"
    # σ берётся максимальная из базы и кандидата
    assert tt.pick_best(base, [_row("a", 0.841, 0.020)])["id"] == "base"


def test_pick_best_takes_highest_among_qualified():
    base = _row("base", 0.80, 0.005)
    cands = [_row("a", 0.82, 0.005), _row("b", 0.85, 0.005), _row("c", 0.803, 0.005)]
    assert tt.pick_best(base, cands)["id"] == "b"
    assert tt.pick_best(base, [])["id"] == "base"


def test_row_to_experiment_roundtrip():
    exp = Experiment("svc", "none", 1500, tail_chars=500, strip_header=False, C=2.0)
    row = {
        "model": exp.model,
        "class_weight": exp.class_weight,
        "max_chars": exp.max_chars,
        "tail_chars": exp.tail_chars,
        "strip_header": exp.strip_header,
        "C": exp.C,
    }
    assert tt.row_to_experiment(row) == exp


def test_row_to_experiment_legacy_row_without_new_columns():
    legacy = {"model": "svc", "class_weight": "none", "max_chars": 2000}
    assert tt.row_to_experiment(legacy) == Experiment("svc", "none", 2000)
    nan = float("nan")
    assert tt.row_to_experiment({**legacy, "tail_chars": nan, "strip_header": nan, "C": nan}) == (
        Experiment("svc", "none", 2000)
    )


def test_stage_grids():
    s1 = tt.truncation_grid()
    assert [e.id for e in s1] == [
        "t2-svc-none-c1000",
        "t2-svc-none-c3000",
        "t2-svc-none-c1500-t500",
        "t2-svc-none-c2000-t1000",
    ]
    best = Experiment("svc", "none", 1500, tail_chars=500)
    s2 = tt.c_grid(best)
    assert [e.C for e in s2] == [0.25, 1.0, 2.0]
    assert all(e.variant == best.variant for e in s2)  # те же признаки — считаются один раз


def test_confirmation_by_control():
    base = _row("base", 0.80, 0.01, control_macro_f1=0.76, control_fold_std=0.012)
    good = _row("f", 0.83, 0.01, control_macro_f1=0.79, control_fold_std=0.012)
    flat = _row("f", 0.83, 0.01, control_macro_f1=0.765, control_fold_std=0.012)
    assert tt.confirmed_by_control(base, good) is True
    assert tt.confirmed_by_control(base, flat) is False  # прирост меньше σ
    assert tt.confirmed_by_control(base, base) is True  # финалист = база: нечего подтверждать


def test_run_tune_synthetic(tmp_path):
    from test_t2_experiment import _cfg

    cfg = _cfg(tmp_path)
    out, pc, summ = tmp_path / "e.csv", tmp_path / "pc.csv", tmp_path / "s.json"
    base = Experiment("svc", "none", 5000)
    res = tt.run_tune(
        cfg,
        out,
        pc,
        summ,
        log=lambda *_: None,
        base=base,
        truncations=[Experiment("svc", "none", 600, tail_chars=200)],
        cs=(0.25, 2.0),
    )
    rows = pd.read_csv(out)
    assert base.id in set(rows["id"]) and any(i.endswith("-nohdr") for i in rows["id"])
    summary = json.loads(summ.read_text())
    assert set(summary) >= {"base", "truncation", "c", "header", "finalist", "confirmed"}
    assert res["finalist"] == summary["finalist"]
    # у финалиста есть обе схемы (подтверждение)
    fin = rows[rows["id"] == summary["finalist"]].iloc[0]
    assert fin["control_macro_f1"] > 0 and fin["main_macro_f1"] > 0


def test_run_tune_resumes_without_recomputing(tmp_path):
    from test_t2_experiment import _cfg

    cfg = _cfg(tmp_path)
    out, pc, summ = tmp_path / "e.csv", tmp_path / "pc.csv", tmp_path / "s.json"
    kw = dict(
        base=Experiment("svc", "none", 5000),
        truncations=[Experiment("svc", "none", 600, tail_chars=200)],
        cs=(2.0,),
    )
    tt.run_tune(cfg, out, pc, summ, log=lambda *_: None, **kw)
    first = pd.read_csv(out)
    logs = []
    tt.run_tune(cfg, out, pc, summ, log=logs.append, **kw)
    assert any("пропуск" in str(m) for m in logs)
    assert not any("fold 0:" in str(m) for m in logs)  # ни одного нового fit
    pd.testing.assert_frame_equal(first, pd.read_csv(out))


def _c(id_, c, f1):
    return {"id": id_, "C": c, "main_macro_f1": f1, "main_fold_std": 0.008}


def test_pick_smallest_within_sigma():
    rows = [_c("a", 0.5, 0.829), _c("b", 1, 0.8396), _c("c", 2, 0.8429), _c("d", 4, 0.8440)]
    # лучший 0.8440; в пределах σ=0.0086: 1 (0.0044), 2, 4 → наименьший C = 1
    assert tt.pick_smallest_within_sigma(rows, 0.0086)["C"] == 1
    # на границе σ включительно
    assert tt.pick_smallest_within_sigma([_c("a", 1, 0.8354), _c("b", 2, 0.844)], 0.0086)["C"] == 1
    # если выше лучшего больше чем на σ — только он сам
    assert tt.pick_smallest_within_sigma([_c("a", 1, 0.80), _c("b", 8, 0.85)], 0.0086)["C"] == 8


def test_big_drops():
    pc = pd.DataFrame(
        {
            "id": ["ref"] * 3 + ["var"] * 3 + ["ref"] * 3 + ["var"] * 3,
            "scheme": ["main"] * 6 + ["control"] * 6,
            "code": ["010.000.000", "020.000.000", "030.000.000"] * 4,
            "f1": [0.9, 0.8, 0.7, 0.7, 0.79, 0.5, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
        }
    )
    d = tt.big_drops(pc, "ref", "var", "main", 0.10)
    assert d == {"010.000.000": -0.2, "030.000.000": -0.2}
    assert tt.big_drops(pc, "ref", "var", "control", 0.10) == {}


def test_ablation_grid():
    g = tt.ablation_grid(Experiment("svc", "none", 2000, C=4.0))
    modes = [e.header_mode for e in g]
    assert modes == [
        "",
        "full",
        "only_type",
        "only_org",
        "only_numdates",
        "no_type",
        "no_org",
        "no_numdates",
    ]
    assert all(e.strip_header and e.C == 4.0 and e.max_chars == 2000 for e in g)


def test_run_tune_c_and_header_synthetic(tmp_path):
    from test_t2_experiment import _cfg

    cfg = _cfg(tmp_path)
    out, pc, summ = tmp_path / "e.csv", tmp_path / "pc.csv", tmp_path / "s2.json"
    base = Experiment("svc", "none", 5000)
    res = tt.run_tune_c_and_header(
        cfg, out, pc, summ, log=lambda *_: None, base=base, extra_cs=(2.0,), sigma=0.0086
    )
    rows = pd.read_csv(out)
    chosen = res["chosen"]
    assert chosen in set(rows["id"])
    hdr_ids = rows.loc[rows["id"].str.contains("nohdr"), "id"]
    assert len(hdr_ids) == 8  # без шапки + full + only×3 + no×3
    for col in ("main_macro_f1", "control_macro_f1"):
        assert rows.loc[rows["id"].isin(hdr_ids), col].notna().all()  # абляция в обеих схемах
    assert set(res) >= {"chosen", "c_candidates", "confirmed", "ablation", "parse_rate", "note"}


def test_resume_rejects_rows_from_limited_run(tmp_path):
    from test_t2_experiment import _cfg

    cfg = _cfg(tmp_path)
    out, pc = tmp_path / "e.csv", tmp_path / "pc.csv"
    exp = Experiment("svc", "none", 5000)
    run = tt._runner(cfg, out, pc, lambda *_: None)
    run([exp], ("main",))
    df = pd.read_csv(out)
    df["n_rows"] = 10  # как будто строка от прогона с --limit
    df.to_csv(out, index=False)
    logs = []
    tt._runner(cfg, out, pc, logs.append)([exp], ("main",))
    assert not any("пропуск" in str(m) for m in logs)  # строка пересчитана
    assert pd.read_csv(out).loc[0, "n_rows"] == 90
