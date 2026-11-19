"""T2: бейзлайны под схемы валидации записи 13 (OOF, срезы по органам, редкие классы).

Правило A: векторайзеры и модели обучаются только на train-части фолда. Признаки считаются
один раз на (max_chars, схема, фолд) и переиспользуются всеми моделями. Сабмишн не пишется.
"""

import dataclasses
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from itgorod.config import T2Config
from itgorod.data import build_t2_cache
from itgorod.t2_cv import (
    build_meta,
    control_folds,
    fit_fold_features,
    main_folds,
    oof_report,
    seen_org_mask,
)

N_SPLITS = 5
MEMORY_LIMIT_MB = 4096
META_MAX_CHARS = 5000  # хеш тела и орган считаются по этому усечению для всех max_chars
CHAR_MAX_FEATURES = 100_000  # char_wb (3,4): (3,5) с 200k признаков не лезет в 4 ГБ у liblinear


class MemoryLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Experiment:
    model: str  # logreg | svc | cnb
    class_weight: str  # none | balanced
    max_chars: int

    @property
    def id(self) -> str:
        return f"t2-{self.model}-{self.class_weight}-c{self.max_chars}"


def default_grid() -> list[Experiment]:
    """5000: все модели и обе настройки class_weight (CNB весов не имеет); 2000 и 10000 —
    только LinearSVC (быстрая модель) для проверки усечения."""
    grid = [Experiment(m, cw, 5000) for m in ("logreg", "svc") for cw in ("none", "balanced")] + [
        Experiment("cnb", "none", 5000)
    ]
    grid += [Experiment("svc", cw, c) for c in (2000, 10000) for cw in ("none", "balanced")]
    return grid


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def check_memory() -> None:
    if peak_rss_mb() > MEMORY_LIMIT_MB:
        raise MemoryLimitExceeded(f"пик RSS {peak_rss_mb():.0f} МиБ > {MEMORY_LIMIT_MB}")


def make_vectorizer(cfg: T2Config, char_max_features: int = CHAR_MAX_FEATURES) -> FeatureUnion:
    word = {**cfg.tfidf, "ngram_range": tuple(cfg.tfidf.get("ngram_range", (1, 1)))}
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(**word, dtype=np.float32)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 4),
                    max_features=char_max_features,
                    min_df=3,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def make_model(exp: Experiment, seed: int):
    cw = None if exp.class_weight == "none" else "balanced"
    if exp.model == "logreg":
        return LogisticRegression(C=4.0, max_iter=200, class_weight=cw, random_state=seed)
    if exp.model == "svc":
        return LinearSVC(C=0.5, class_weight=cw, random_state=seed)
    if exp.model == "cnb":
        return ComplementNB(alpha=0.3)
    raise ValueError(exp.model)


def load_train(cfg: T2Config, max_chars: int) -> pd.DataFrame:
    path = build_t2_cache(dataclasses.replace(cfg, max_chars=max_chars), "train", smoke=False)
    return pd.read_parquet(path)


def load_meta(cfg: T2Config) -> pd.DataFrame:
    """code, org_norm, body_hash; кэшируется (орган и хеш не зависят от grid max_chars)."""
    path = cfg.paths.processed_dir / f"t2_meta_c{META_MAX_CHARS}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = load_train(cfg, META_MAX_CHARS)
    meta = pd.concat([df[["code"]], build_meta(df["text"].tolist())], axis=1)
    meta.to_parquet(path, index=False)
    return meta


def make_schemes(meta: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    codes = meta["code"].to_numpy()
    return {
        "main": main_folds(codes, meta["body_hash"], N_SPLITS, seed),
        "control": control_folds(codes, meta["body_hash"], meta["org_norm"], N_SPLITS, seed),
    }


def run_experiments(
    cfg: T2Config,
    grid: list[Experiment],
    out_csv: Path,
    per_class_csv: Path,
    limit: int | None = None,
    log=print,
) -> pd.DataFrame:
    meta = load_meta(cfg)
    if limit:
        meta = meta.iloc[:limit].reset_index(drop=True)
    y = meta["code"].to_numpy()
    schemes = make_schemes(meta, cfg.seed)
    seen = {s: seen_org_mask(meta["org_norm"], f) for s, f in schemes.items()}
    rows, per_class = [], []
    for max_chars in sorted({e.max_chars for e in grid}):
        exps = [e for e in grid if e.max_chars == max_chars]
        frame = load_train(cfg, max_chars)
        texts = frame["text"].to_numpy()
        del frame
        if limit:
            texts = texts[:limit]
        preds = {(e.id, s): np.empty(len(y), dtype=object) for e in exps for s in schemes}
        secs = {e.id: 0.0 for e in exps}
        for scheme, folds in schemes.items():
            for k in range(N_SPLITS):
                va = np.flatnonzero(folds == k)
                tr = np.flatnonzero(folds != k)
                t0 = time.time()
                xtr, xva = fit_fold_features(make_vectorizer(cfg), texts[tr], texts[va])
                check_memory()
                feat_s = time.time() - t0
                for e in exps:
                    t1 = time.time()
                    model = make_model(e, cfg.seed).fit(xtr, y[tr])
                    preds[(e.id, scheme)][va] = model.predict(xva)
                    secs[e.id] += time.time() - t1 + feat_s / len(exps)
                    check_memory()
                log(f"{max_chars} {scheme} fold {k}: {feat_s:.0f}с, RSS {peak_rss_mb():.0f}")
        for e in exps:
            row = {
                "id": e.id,
                "model": e.model,
                "class_weight": e.class_weight,
                "max_chars": e.max_chars,
                "seconds": round(secs[e.id]),
                "peak_rss_mb": round(peak_rss_mb()),
                "n_rows": len(y),
            }
            for scheme, folds in schemes.items():
                p = preds[(e.id, scheme)]
                rep = oof_report(y, p, seen[scheme])
                fold_f1 = [
                    oof_report(y[folds == k], p[folds == k], seen[scheme][folds == k])["macro_f1"]
                    for k in range(N_SPLITS)
                ]
                row.update(
                    {
                        f"{scheme}_macro_f1": round(rep["macro_f1"], 4),
                        f"{scheme}_fold_std": round(float(np.std(fold_f1)), 4),
                        f"{scheme}_f1_seen": round(rep["macro_f1_seen"], 4),
                        f"{scheme}_f1_unseen": round(rep["macro_f1_unseen"], 4),
                        f"{scheme}_f1_rare": round(rep["macro_f1_rare"], 4),
                        f"{scheme}_n_unseen": rep["n_unseen"],
                    }
                )
                pc = rep["per_class"].assign(id=e.id, scheme=scheme)
                per_class.append(pc)
            rows.append(row)
        # промежуточная запись после каждого max_chars: падение не теряет готовые строки
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        pd.concat(per_class, ignore_index=True).to_csv(per_class_csv, index=False)
    return pd.DataFrame(rows)
