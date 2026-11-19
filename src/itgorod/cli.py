"""CLI `itgorod`: тонкая обёртка над библиотекой (логика — в t1/t2/data/validation)."""

import argparse
import sys

import pandas as pd

from itgorod.config import REPO_ROOT, load_t1_config, load_t2_config
from itgorod.data import build_t2_cache, read_t1, read_t2_codes
from itgorod.t1 import TARGET, run_t1
from itgorod.t1_experiment import run_t1_experiment
from itgorod.t2 import run_t2_smoke
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
        print("полный прогон — на этапе T2 (пока доступен только --smoke)")
        return 2
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
