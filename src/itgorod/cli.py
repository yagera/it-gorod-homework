"""CLI `itgorod`: тонкая обёртка над библиотекой (логика — в t1/t2/data/validation)."""

import argparse
import sys

import pandas as pd

from itgorod.config import REPO_ROOT, load_t1_config, load_t2_config
from itgorod.data import build_t2_cache, read_t1, read_t2_codes
from itgorod.t1 import TARGET, run_t1
from itgorod.t1_experiment import run_t1_experiment
from itgorod.t2 import run_t2_full, run_t2_smoke
from itgorod.validation import (
    LeakageError,
    SubmissionError,
    assert_features_for_year,
    assert_folds_ordered,
    assert_no_time_leakage,
    read_submission_csv,
    split_fold,
    validate_t1_submission,
    validate_t2_submission,
)


def _cmd_t1_predict(args) -> int:
    print(f"written: {run_t1(load_t1_config(args.config))}")
    return 0


def _cmd_t1_experiment(args) -> int:
    cfg = load_t1_config(args.config)
    run_t1_experiment(cfg)
    print(f"written: {cfg.experiment.output}, {cfg.experiment.grid_output}")
    return 0


def _cmd_t2_prepare(args) -> int:
    cfg = load_t2_config(args.config)
    for split in ("train", "test"):
        print(f"cache: {build_t2_cache(cfg, split, smoke=args.smoke, force=args.force)}")
    return 0


def _cmd_t2_train_predict(args) -> int:
    cfg = load_t2_config(args.config)
    if not args.smoke:
        print(f"written: {run_t2_full(cfg, force=args.force)}")
        return 0
    print(f"written (smoke): {run_t2_smoke(cfg, force=args.force)}")
    return 0


def _cmd_t2_experiment(args) -> int:
    from itgorod.t2_experiment import MemoryLimitExceeded, default_grid, run_experiments

    cfg = load_t2_config(args.config)
    out = REPO_ROOT / "outputs"
    try:
        run_experiments(
            cfg, default_grid(), out / "t2_experiments.csv", out / "t2_per_class.csv", args.limit
        )
    except MemoryLimitExceeded as exc:
        print(f"STOP: {exc}")
        return 3
    print(f"written: {out / 't2_experiments.csv'}, {out / 't2_per_class.csv'}")
    return 0


def _cmd_t2_tune(args) -> int:
    from itgorod.t2_experiment import MemoryLimitExceeded
    from itgorod.t2_tune import run_tune

    cfg = load_t2_config(args.config)
    out = REPO_ROOT / "outputs"
    try:
        summary = run_tune(
            cfg, out / "t2_experiments.csv", out / "t2_per_class.csv", out / "t2_tune_summary.json"
        )
    except MemoryLimitExceeded as exc:
        print(f"STOP: {exc}")
        return 3
    print(f"written: {out / 't2_tune_summary.json'}; финалист {summary['finalist']}")
    return 0


def _cmd_t2_tune2(args) -> int:
    from itgorod.t2_experiment import MemoryLimitExceeded
    from itgorod.t2_tune import run_tune_c_and_header

    cfg = load_t2_config(args.config)
    out = REPO_ROOT / "outputs"
    try:
        s = run_tune_c_and_header(
            cfg,
            out / "t2_experiments.csv",
            out / "t2_per_class.csv",
            out / "t2_tune2_summary.json",
        )
    except MemoryLimitExceeded as exc:
        print(f"STOP: {exc}")
        return 3
    print(f"written: {out / 't2_tune2_summary.json'}; выбран {s['chosen']}")
    return 0


def _cmd_t2_stability(args) -> int:
    from itgorod.t2_experiment import MemoryLimitExceeded
    from itgorod.t2_tune import run_date_stability

    cfg = load_t2_config(args.config)
    out = REPO_ROOT / "outputs"
    try:
        run_date_stability(
            cfg,
            out / "t2_experiments.csv",
            out / "t2_per_class.csv",
            out / "t2_stability_summary.json",
        )
    except MemoryLimitExceeded as exc:
        print(f"STOP: {exc}")
        return 3
    print(f"written: {out / 't2_stability_summary.json'}")
    return 0


def _cmd_t2_swapdate(args) -> int:
    from itgorod.t2_experiment import MemoryLimitExceeded
    from itgorod.t2_tune import run_swap_date_eval

    cfg = load_t2_config(args.config)
    out = REPO_ROOT / "outputs"
    try:
        run_swap_date_eval(
            cfg,
            out / "t2_experiments.csv",
            out / "t2_per_class.csv",
            out / "t2_swapdate_summary.json",
        )
    except MemoryLimitExceeded as exc:
        print(f"STOP: {exc}")
        return 3
    print(f"written: {out / 't2_swapdate_summary.json'}")
    return 0


def _cmd_t2_embed(args) -> int:
    from itgorod.embeddings import HFEncoder, embed_corpus

    cfg = load_t2_config(args.config)
    encoder = HFEncoder(args.model, max_tokens=args.max_tokens)
    paths = embed_corpus(
        cfg, encoder, args.model, args.max_tokens, REPO_ROOT / "data" / "cache", args.batch_size
    )
    for p in paths:
        print(f"cache: {p}")
    return 0


def _cmd_t2_oof_svc(args) -> int:
    from itgorod.t2 import FULL_EXPERIMENT
    from itgorod.t2_embed_eval import compute_svc_oof

    cfg = load_t2_config(args.config)
    compute_svc_oof(cfg, FULL_EXPERIMENT, REPO_ROOT / "outputs")
    return 0


def _cmd_t2_embed_eval(args) -> int:
    from itgorod.t2 import FULL_EXPERIMENT
    from itgorod.t2_embed_eval import compute_svc_oof, run_embed_eval
    from itgorod.t2_experiment import load_meta, make_schemes

    cfg = load_t2_config(args.config)
    meta = load_meta(cfg)
    out = REPO_ROOT / "outputs"
    svc = compute_svc_oof(cfg, FULL_EXPERIMENT, out)
    res = run_embed_eval(
        args.model,
        args.max_tokens,
        meta["code"].to_numpy(),
        meta["org_norm"],
        make_schemes(meta, cfg.seed),
        svc,
        REPO_ROOT / "data" / "cache",
        out,
        max_iter=args.max_iter,
    )
    print(f"written: outputs/t2_embed_eval_*; {res['model']}")
    return 0


def _cmd_validate(args) -> int:
    try:
        return _validate(args)
    except FileNotFoundError as exc:
        # код 2 — «не удалось проверить»; код 1 зарезервирован за нарушениями формата
        print(f"ERROR: file not found: {exc.filename}")
        return 2


def _validate(args) -> int:
    sub = read_submission_csv(args.csv)
    try:
        if args.task == "t1":
            cfg = load_t1_config(args.config or "configs/t1.yaml")
            validate_t1_submission(sub, read_submission_csv(cfg.paths.sample))
        else:
            cfg = load_t2_config(args.config or "configs/t2.yaml")
            validate_t2_submission(
                sub, read_submission_csv(cfg.paths.sample), read_t2_codes(cfg.paths.train)
            )
    except SubmissionError as exc:
        print(f"INVALID {args.csv}:")
        for error in exc.errors:
            print(f"  - {error}")
        return 1
    print(f"OK {args.csv}: {len(sub)} rows")
    return 0


def _cmd_check_time_split(args) -> int:
    cfg = load_t1_config(args.config)
    train, test = read_t1(cfg.paths.train), read_t1(cfg.paths.test)
    try:
        assert_folds_ordered(cfg.folds)
        for fold in cfg.folds:
            tr, va = split_fold(train, fold)
            assert_no_time_leakage(tr, va)
            print(f"fold <={fold.train_max_year} -> {fold.valid_year}: OK")
        assert_features_for_year(pd.concat([train, test]), cfg.target_year, TARGET)
        print(f"features for {cfg.target_year}: OK")
    except LeakageError as exc:
        print("LEAKAGE:")
        for error in exc.errors:
            print(f"  - {error}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="itgorod")
    sub = parser.add_subparsers(dest="command", required=True)

    t1 = sub.add_parser("t1").add_subparsers(dest="action", required=True)
    p = t1.add_parser("predict")
    p.add_argument("--config", default="configs/t1.yaml")
    p.set_defaults(func=_cmd_t1_predict)
    p = t1.add_parser("experiment")
    p.add_argument("--config", default="configs/t1.yaml")
    p.set_defaults(func=_cmd_t1_experiment)

    t2 = sub.add_parser("t2").add_subparsers(dest="action", required=True)
    for name, func in (("prepare", _cmd_t2_prepare), ("train-predict", _cmd_t2_train_predict)):
        p = t2.add_parser(name)
        p.add_argument("--config", default="configs/t2.yaml")
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--force", action="store_true")
        p.set_defaults(func=func)

    p = t2.add_parser("experiment")
    p.add_argument("--config", default="configs/t2.yaml")
    p.add_argument("--limit", type=int, default=None, help="первые N строк train (отладка)")
    p.set_defaults(func=_cmd_t2_experiment)

    p = t2.add_parser("tune")
    p.add_argument("--config", default="configs/t2.yaml")
    p.set_defaults(func=_cmd_t2_tune)

    p = t2.add_parser("tune2")
    p.add_argument("--config", default="configs/t2.yaml")
    p.set_defaults(func=_cmd_t2_tune2)

    p = t2.add_parser("stability")
    p.add_argument("--config", default="configs/t2.yaml")
    p.set_defaults(func=_cmd_t2_stability)

    p = t2.add_parser("swapdate")
    p.add_argument("--config", default="configs/t2.yaml")
    p.set_defaults(func=_cmd_t2_swapdate)

    p = t2.add_parser("embed")
    p.add_argument("--config", default="configs/t2.yaml")
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=_cmd_t2_embed)

    p = t2.add_parser("oof-svc")
    p.add_argument("--config", default="configs/t2.yaml")
    p.set_defaults(func=_cmd_t2_oof_svc)

    p = t2.add_parser("embed-eval")
    p.add_argument("--config", default="configs/t2.yaml")
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-iter", type=int, default=300, help="итерации LogisticRegression")
    p.set_defaults(func=_cmd_t2_embed_eval)

    p = sub.add_parser("validate")
    p.add_argument("task", choices=["t1", "t2"])
    p.add_argument("csv")
    p.add_argument("--config")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("check-time-split")
    p.add_argument("--config", default="configs/t1.yaml")
    p.set_defaults(func=_cmd_check_time_split)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
