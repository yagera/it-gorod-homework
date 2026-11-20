"""T2: бейзлайны под схемы валидации записи 13 (OOF, срезы по органам, редкие классы).

Правило A: векторайзеры и модели обучаются только на train-части фолда. Признаки считаются
один раз на (max_chars, схема, фолд) и переиспользуются всеми моделями. Сабмишн не пишется.
"""

import dataclasses
import json
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
    fetch_date_pool,
    fit_fold_features,
    group_ids,
    main_folds,
    oof_report,
    seen_org_mask,
    swap_valid_dates,
)
from itgorod.t2_recon import HEADER_PARTS, header_part_text, split_header

N_SPLITS = 5


def _ram_limit_mb(share: float = 0.75) -> float:
    """75% физической RAM из /proc/meminfo (MemTotal)."""
    with open("/proc/meminfo") as f:
        kb = int(next(line for line in f if line.startswith("MemTotal")).split()[1])
    return kb / 1024 * share


LOGREG_MAX_ITER = 200
MEMORY_LIMIT_MB = _ram_limit_mb()
META_MAX_CHARS = 5000  # хеш тела и орган считаются по этому усечению для всех max_chars
CHAR_MAX_FEATURES = 50_000  # пик памяти — float64-копия матрицы в LinearSVC, см. записи 14


class MemoryLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Experiment:
    model: str  # logreg | svc | cnb
    class_weight: str  # none | balanced
    max_chars: int
    tail_chars: int = 0  # >0: «голова + хвост» (max_chars первых + tail_chars последних)
    strip_header: bool = False  # без шапки до «Текст документа»
    C: float = 0.5  # только LinearSVC
    # абляция шапки: перед телом (без шапки, `strip_header=True`) ставятся только выбранные части
    # шапки — only_<часть> | no_<часть> | full, части: type, org, numdates (см. `t2_recon`)
    header_mode: str = ""
    # возмущение valid-текстов при оценке (модель обучается как обычно): swap_fetch_date —
    # дата выгрузки в valid заменяется случайной датой из train-части фолда
    perturb: str = ""

    def __post_init__(self):
        if self.perturb not in ("", "swap_fetch_date"):
            raise ValueError(f"неизвестное возмущение: {self.perturb!r}")
        if self.perturb and (self.header_mode or self.strip_header or self.tail_chars):
            raise ValueError("swap_fetch_date поддержан только для текстов с исходной шапкой")

    @property
    def group_key(self) -> tuple:
        """Признаки общие внутри группы: вариант текста и возмущение valid."""
        return (*self.variant, self.perturb)

    @property
    def variant(self) -> tuple[int, int, bool, str]:
        """Вариант текста: признаки считаются один раз на (вариант, схема, фолд)."""
        return (self.max_chars, self.tail_chars, self.strip_header, self.header_mode)

    @property
    def id(self) -> str:
        tail = f"-t{self.tail_chars}" if self.tail_chars else ""
        hdr = "-nohdr" if self.strip_header else ""
        c = f"-C{self.C:g}" if self.C != 0.5 else ""
        mode = f"-hdr-{self.header_mode}" if self.header_mode else ""
        swap = "-swapdate" if self.perturb else ""
        return f"t2-{self.model}-{self.class_weight}-c{self.max_chars}{tail}{hdr}{c}{mode}{swap}"


def default_grid() -> list[Experiment]:
    """max_chars ∈ {2000, 5000}; LinearSVC / LogisticRegression / ComplementNB без весов;
    `balanced` только у LinearSVC и только на 5000 (на 2000 не проверялся — сетка ограничена)."""
    grid = [Experiment(m, "none", c) for c in (2000, 5000) for m in ("svc", "logreg", "cnb")]
    grid.append(Experiment("svc", "balanced", 5000))
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
        return LogisticRegression(
            C=4.0, max_iter=LOGREG_MAX_ITER, class_weight=cw, random_state=seed
        )
    if exp.model == "svc":
        return LinearSVC(C=exp.C, class_weight=cw, random_state=seed)
    if exp.model == "cnb":
        return ComplementNB(alpha=0.3)
    raise ValueError(exp.model)


def load_train(
    cfg: T2Config, max_chars: int, tail_chars: int = 0, strip_header: bool = False
) -> pd.DataFrame:
    variant = dataclasses.replace(
        cfg, max_chars=max_chars, tail_chars=tail_chars, strip_header=strip_header
    )
    return pd.read_parquet(build_t2_cache(variant, "train", smoke=False))


def header_mode_parts(mode: str) -> tuple[str, ...]:
    """Какие части шапки остаются: `only_X`, `no_X` или `full`."""
    if mode == "full":
        return HEADER_PARTS
    kind, _, part = mode.partition("_")
    if kind == "only" and part in HEADER_PARTS:
        return (part,)
    if kind == "no" and part in HEADER_PARTS:
        return tuple(p for p in HEADER_PARTS if p != part)
    raise ValueError(f"неизвестный режим шапки: {mode!r}")


def load_variant_texts(cfg: T2Config, exp: Experiment) -> np.ndarray:
    """Тексты варианта. Без `header_mode` — как раньше; иначе тело (без шапки, окно
    `max_chars`) с префиксом из выбранных частей шапки, разобранной из исходного текста
    документа. Разбор детерминирован по самому документу (ничего не обучается)."""
    if not exp.header_mode:
        return load_train(cfg, *exp.variant[:3])["text"].to_numpy()
    parts = header_mode_parts(exp.header_mode)
    body = load_train(cfg, exp.max_chars, exp.tail_chars, True)["text"].to_numpy()
    head = load_train(cfg, exp.max_chars, 0, False)["text"].to_numpy()
    out = np.empty(len(body), dtype=object)
    for i, (h, b) in enumerate(zip(head, body, strict=True)):
        hdr = split_header(h)
        prefix = " ".join(t for t in (header_part_text(hdr, p) for p in parts) if t)
        out[i] = f"{prefix} {b}" if prefix else b
    return out


def header_parse_rate(cfg: T2Config) -> dict:
    """Доля документов train, у которых шапка разобрана полностью (по кэшу `max_chars`=2000)."""
    head = load_train(cfg, 2000)["text"].to_numpy()
    parsed = sum(split_header(t)["parsed"] for t in head)
    return {"n": len(head), "parsed": int(parsed), "share": parsed / len(head)}


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
    scheme_names: tuple[str, ...] = ("main", "control"),
    append: bool = False,
    extra_schemes: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """`scheme_names` — какие схемы считать; `append` — дописать к существующим CSV (строки с
    теми же id сливаются: новые значения заменяют старые, остальные колонки сохраняются).
    `extra_schemes` — дополнительные схемы фолдов по строкам (имя → массив номеров фолдов),
    считаются, если имя есть в `scheme_names`. Одинаковые id в сетке считаются один раз."""
    grid = list({e.id: e for e in grid}.values())  # одинаковые id считаются один раз
    meta = load_meta(cfg)
    if limit:
        meta = meta.iloc[:limit].reset_index(drop=True)
    y = meta["code"].to_numpy()
    all_schemes = {**make_schemes(meta, cfg.seed), **(extra_schemes or {})}
    schemes = {k: v for k, v in all_schemes.items() if k in scheme_names}
    for name, f in schemes.items():
        sizes = np.bincount(group_ids(meta["body_hash"], meta["org_norm"]))
        log(
            f"схема {name}: фолды {np.bincount(f).tolist()}, "
            f"крупнейшая группа {sizes.max()} ({sizes.max() / len(f):.1%} строк)"
        )
        if name == "control" and sizes.max() > len(f) / N_SPLITS:
            log("ВНИМАНИЕ: крупнейшая группа больше 1/n_splits — контрольная схема искажена")
    seen = {s: seen_org_mask(meta["org_norm"], f) for s, f in schemes.items()}
    rows, per_class = [], []
    for gkey in dict.fromkeys(e.group_key for e in grid):
        variant = gkey[:4]
        exps = [e for e in grid if e.group_key == gkey]
        texts = load_variant_texts(cfg, exps[0])
        if limit:
            texts = texts[:limit]
        preds = {(e.id, s): np.empty(len(y), dtype=object) for e in exps for s in schemes}
        secs = {e.id: 0.0 for e in exps}
        not_converged = {(e.id, s): 0 for e in exps for s in schemes}
        perturbed = {(e.id, s): 0 for e in exps for s in schemes}
        for scheme, folds in schemes.items():
            for k in np.unique(folds):
                va = np.flatnonzero(folds == k)
                tr = np.flatnonzero(folds != k)
                t0 = time.time()
                va_texts = texts[va]
                n_changed = 0
                if exps[0].perturb:  # группа целиком с возмущением valid (train не меняется)
                    va_texts, n_changed = swap_valid_dates(
                        va_texts, fetch_date_pool(texts[tr]), (cfg.seed, int(k))
                    )
                xtr, xva = fit_fold_features(make_vectorizer(cfg), texts[tr], va_texts)
                check_memory()
                feat_s = time.time() - t0
                for e in exps:
                    t1 = time.time()
                    model = make_model(e, cfg.seed).fit(xtr, y[tr])
                    preds[(e.id, scheme)][va] = model.predict(xva)
                    secs[e.id] += time.time() - t1 + feat_s / len(exps)
                    perturbed[(e.id, scheme)] += n_changed
                    if e.model == "logreg":
                        not_converged[(e.id, scheme)] += int(model.n_iter_.max() >= LOGREG_MAX_ITER)
                    fit_s = time.time() - t1
                    log(f"{e.id} {scheme} fold {k}: {fit_s:.0f}с, RSS {peak_rss_mb():.0f}")
                    check_memory()
                del xtr, xva
                log(f"{variant} {scheme} fold {k} готов: признаки {feat_s:.0f}с")
        for e in exps:
            row = {
                "id": e.id,
                "model": e.model,
                "class_weight": e.class_weight,
                "max_chars": e.max_chars,
                "tail_chars": e.tail_chars,
                "strip_header": e.strip_header,
                "C": e.C,
                "header_mode": e.header_mode,
                "seconds": round(secs[e.id]),
                "peak_rss_mb": round(peak_rss_mb()),
                "n_rows": len(y),
            }
            for scheme, folds in schemes.items():
                p = preds[(e.id, scheme)]
                rep = oof_report(y, p, seen[scheme])
                fold_f1 = [
                    oof_report(y[folds == k], p[folds == k], seen[scheme][folds == k])["macro_f1"]
                    for k in np.unique(folds)
                ]
                row.update(
                    {
                        f"{scheme}_not_converged_folds": not_converged[(e.id, scheme)]
                        if e.model == "logreg"
                        else "",
                        f"{scheme}_macro_f1": round(rep["macro_f1"], 4),
                        f"{scheme}_fold_std": round(float(np.std(fold_f1)), 4),
                        f"{scheme}_fold_f1": json.dumps([round(float(x), 4) for x in fold_f1]),
                        f"{scheme}_f1_seen": round(rep["macro_f1_seen"], 4),
                        f"{scheme}_f1_unseen": round(rep["macro_f1_unseen"], 4),
                        f"{scheme}_f1_rare": round(rep["macro_f1_rare"], 4),
                        f"{scheme}_n_unseen": rep["n_unseen"],
                        **(
                            {f"{scheme}_n_perturbed": perturbed[(e.id, scheme)]}
                            if e.perturb
                            else {}
                        ),
                    }
                )
                pc = rep["per_class"].assign(id=e.id, scheme=scheme)
                per_class.append(pc)
            rows.append(row)
        # промежуточная запись после каждого варианта текста: падение не теряет готовые строки
        _write(out_csv, per_class_csv, rows, per_class, append)
    return pd.DataFrame(rows)


def _write(out_csv: Path, per_class_csv: Path, rows: list, per_class: list, append: bool) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    pc = pd.concat(per_class, ignore_index=True)
    if append and out_csv.exists():
        old = pd.read_csv(out_csv)
        # слияние по id: новые значения заменяют старые, колонки других схем сохраняются
        order = list(old["id"]) + [i for i in result["id"] if i not in set(old["id"])]
        result = result.drop_duplicates("id", keep="last")
        merged = result.set_index("id").combine_first(old.set_index("id"))
        result = merged.loc[order].reset_index()
        result = result.fillna(
            {"tail_chars": 0, "strip_header": False, "C": 0.5, "header_mode": ""}
        )
        result = result.astype({"tail_chars": int, "strip_header": bool})
    if append and per_class_csv.exists():
        old_pc = pd.read_csv(per_class_csv, dtype={"code": str})
        key = set(zip(pc["id"], pc["scheme"], strict=True))
        keep = [(i, sc) not in key for i, sc in zip(old_pc["id"], old_pc["scheme"], strict=True)]
        pc = pd.concat([old_pc[keep], pc], ignore_index=True)
    result.to_csv(out_csv, index=False)
    pc.to_csv(per_class_csv, index=False)
