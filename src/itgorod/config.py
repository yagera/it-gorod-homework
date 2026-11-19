"""YAML-конфиги → dataclass. Неизвестные ключи отвергаются."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Fold:
    train_max_year: int
    valid_year: int


@dataclass(frozen=True)
class T1Paths:
    train: Path
    test: Path
    sample: Path
    output: Path


@dataclass(frozen=True)
class T1Experiment:
    outlier_threshold: float
    w_grid: list[float]
    phi_grid: list[float]
    small_threshold: float
    n_bootstrap: int
    output: Path
    grid_output: Path


@dataclass(frozen=True)
class T1Config:
    seed: int
    paths: T1Paths
    folds: list[Fold]
    target_year: int
    fallback: str
    experiment: T1Experiment | None = None


@dataclass(frozen=True)
class T2Paths:
    train: Path
    test: Path
    sample: Path
    processed_dir: Path
    output: Path
    smoke_output: Path


@dataclass(frozen=True)
class T2Config:
    seed: int
    paths: T2Paths
    batch_size: int
    max_chars: int
    smoke_n_train_rows: int
    tfidf: dict
    logreg: dict


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load(path: str | Path, allowed: set[str]) -> dict[str, Any]:
    with open(_resolve(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _check_keys(raw, allowed, "<root>")
    return raw


def _check_keys(d: dict, allowed: set[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"Unknown config key(s) {sorted(unknown)} in {where}")


def _paths(d: dict, keys: set[str], where: str) -> dict[str, Path]:
    _check_keys(d, keys, where)
    return {k: _resolve(v) for k, v in d.items()}


def _folds(items: list[dict]) -> list[Fold]:
    folds = []
    for i, item in enumerate(items):
        _check_keys(item, {"train_max_year", "valid_year"}, f"split.folds[{i}]")
        folds.append(Fold(**item))
    return folds


def load_t1_config(path: str | Path) -> T1Config:
    raw = _load(path, {"seed", "paths", "split", "target_year", "placeholder", "experiment"})
    _check_keys(raw["split"], {"folds"}, "split")
    _check_keys(raw["placeholder"], {"fallback"}, "placeholder")
    path_keys = {"train", "test", "sample", "output"}
    return T1Config(
        seed=raw["seed"],
        paths=T1Paths(**_paths(raw["paths"], path_keys, "paths")),
        folds=_folds(raw["split"]["folds"]),
        target_year=raw["target_year"],
        fallback=raw["placeholder"]["fallback"],
        experiment=_experiment(raw["experiment"]) if "experiment" in raw else None,
    )


def _experiment(d: dict) -> T1Experiment:
    keys = {f for f in T1Experiment.__dataclass_fields__}
    _check_keys(d, keys, "experiment")
    missing = keys - set(d)
    if missing:
        raise ValueError(f"Missing config key(s) {sorted(missing)} in experiment")
    return T1Experiment(
        outlier_threshold=float(d["outlier_threshold"]),
        w_grid=[float(x) for x in d["w_grid"]],
        phi_grid=[float(x) for x in d["phi_grid"]],
        small_threshold=float(d["small_threshold"]),
        n_bootstrap=int(d["n_bootstrap"]),
        output=_resolve(d["output"]),
        grid_output=_resolve(d["grid_output"]),
    )


def load_t2_config(path: str | Path) -> T2Config:
    raw = _load(path, {"seed", "paths", "read", "smoke", "tfidf", "logreg"})
    _check_keys(raw["read"], {"batch_size", "max_chars"}, "read")
    _check_keys(raw["smoke"], {"n_train_rows"}, "smoke")
    path_keys = {"train", "test", "sample", "processed_dir", "output", "smoke_output"}
    return T2Config(
        seed=raw["seed"],
        paths=T2Paths(**_paths(raw["paths"], path_keys, "paths")),
        batch_size=raw["read"]["batch_size"],
        max_chars=raw["read"]["max_chars"],
        smoke_n_train_rows=raw["smoke"]["n_train_rows"],
        tfidf=raw["tfidf"],
        logreg=raw["logreg"],
    )
