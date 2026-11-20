"""T2: OOF-оценка эмбеддингов (LogisticRegression) и бленд с TF-IDF + LinearSVC.

Правило A: энкодер не видит меток (эмбеддинги считаются заранее только по текстам); scaler и
LogisticRegression обучаются только на train-части фолда. Бленд — взвешенная сумма построчно
стандартизованных оценок двух моделей (OOF): один параметр веса, выбор по сумме Macro-F1 двух
схем. Выбор веса по тем же OOF — оценка выбора оптимистична.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from itgorod.config import T2Config
from itgorod.embeddings import EmbeddingSpec, cache_path, model_slug
from itgorod.t2_cv import fit_fold_features, oof_report, seen_org_mask

SIGMA = 0.0086  # σ по фолдам базовой конфигурации (C=0,5, основная схема; записи 14–17)
BLEND_WEIGHTS = tuple(float(w) for w in np.round(np.arange(0, 1.01, 0.1), 1))
FLOOR_MARGIN = 10.0  # оценка класса, которого нет в train-части фолда: минимум строки − margin


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    """Построчная стандартизация оценок классов (постоянная строка → нули)."""
    s = np.asarray(scores, dtype=np.float64)
    std = s.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (s - s.mean(axis=1, keepdims=True)) / std


def blend_predict(a: np.ndarray, b: np.ndarray, w: float, classes: np.ndarray) -> np.ndarray:
    """Класс по (1 − w)·z(a) + w·z(b); w=0 — только первая модель, w=1 — только вторая."""
    s = (1 - w) * zscore_rows(a) + w * zscore_rows(b)
    return np.asarray(classes)[s.argmax(axis=1)]


def place_scores(fold_scores: np.ndarray, model_classes, classes: np.ndarray) -> np.ndarray:
    """Раскладывает оценки модели по общему списку классов; отсутствующие классы получают
    минимум строки минус `FLOOR_MARGIN` (их предсказать нельзя)."""
    out = np.tile(fold_scores.min(axis=1, keepdims=True) - FLOOR_MARGIN, (1, len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        out[:, idx[c]] = fold_scores[:, j]
    return out


def oof_logits(X: np.ndarray, y, folds: np.ndarray, C: float = 1.0, max_iter: int = 300):
    """OOF-логиты LogisticRegression на эмбеддингах: scaler и модель — только на train-части."""
    y = np.asarray(y)
    classes = np.array(sorted(set(y)))
    out = np.zeros((len(y), len(classes)), dtype=np.float32)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(C=C, max_iter=max_iter).fit(scaler.transform(X[tr]), y[tr])
        dec = model.decision_function(scaler.transform(X[va]))
        if dec.ndim == 1:  # два класса в train-части: sklearn отдаёт один столбец
            dec = np.stack([-dec, dec], axis=1)
        out[va] = place_scores(dec, model.classes_, classes)
    return out, classes


def pick_blend_weight(rows: list[tuple], sigma: float) -> tuple[float, dict]:
    """rows — (w, Macro-F1 основной, Macro-F1 контрольной); w=0 — базовая модель. Лучший w по
    сумме двух схем; если средний по двум схемам прирост относительно w=0 не превышает σ —
    прироста нет, берётся w=0."""
    base = next(r for r in rows if r[0] == 0.0)
    best = max(rows, key=lambda r: r[1] + r[2])
    gain = ((best[1] - base[1]) + (best[2] - base[2])) / 2
    improved = bool(best[0] != 0.0 and gain > sigma)
    return (best[0] if improved else 0.0), {"best_w": best[0], "gain": gain, "improved": improved}


def load_embeddings(cache_dir: str | Path, spec: EmbeddingSpec) -> np.ndarray:
    path = cache_path(cache_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"нет кэша эмбеддингов: {path.name}")
    return np.load(path).astype(np.float32)


def oof_tfidf_scores(cfg: T2Config, folds: np.ndarray, exp, classes: np.ndarray) -> np.ndarray:
    """OOF decision_function TF-IDF + LinearSVC (конфиг `exp`, правило A: fit по train-части)."""
    from itgorod.t2_experiment import load_meta, load_variant_texts, make_model, make_vectorizer

    texts = load_variant_texts(cfg, exp)
    y = load_meta(cfg)["code"].to_numpy()
    out = np.zeros((len(y), len(classes)), dtype=np.float32)
    for k in np.unique(folds):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        xtr, xva = fit_fold_features(make_vectorizer(cfg), texts[tr], texts[va])
        model = make_model(exp, cfg.seed).fit(xtr, y[tr])
        out[va] = place_scores(model.decision_function(xva), model.classes_, classes)
    return out


def report_row(id_: str, y, preds: dict, orgs, folds: dict) -> dict:
    """Строка результатов: Macro-F1 и F1 по редким классам для обеих схем."""
    row = {"id": id_}
    for scheme, p in preds.items():
        seen = seen_org_mask(orgs, folds[scheme])
        rep = oof_report(y, p, seen)
        row[f"{scheme}_macro_f1"] = round(rep["macro_f1"], 4)
        row[f"{scheme}_f1_rare"] = round(rep["macro_f1_rare"], 4)
    return row


def blend_curve(
    svc: dict,
    lr: dict,
    y,
    classes,
    orgs,
    folds: dict,
    ws=None,
) -> pd.DataFrame:
    """Macro-F1 бленда по сетке весов (0, 0.1 … 1) для обеих схем: svc[scheme], lr[scheme] —
    OOF-оценки."""
    ws = BLEND_WEIGHTS if ws is None else ws
    rows = []
    for w in ws:
        preds = {s: blend_predict(svc[s], lr[s], float(w), classes) for s in svc}
        r = report_row(f"w={w:.1f}", y, preds, orgs, folds)
        r["w"] = float(w)
        rows.append(r)
    return pd.DataFrame(rows)


def run_embed_eval(
    model: str,
    max_tokens: int,
    y,
    orgs,
    folds: dict,
    svc_scores: dict,
    cache_dir: str | Path,
    out_dir: str | Path,
    log=print,
    sigma: float = SIGMA,
    C: float = 1.0,
    max_iter: int = 300,
) -> dict:
    """OOF-оценка эмбеддингов (LogisticRegression) и бленда с TF-IDF SVC в обеих схемах и обоих
    режимах шапки. `svc_scores[схема]` — OOF decision_function базовой модели (строки в порядке
    train). Выбор веса по сумме двух схем; прирост меньше σ — прироста нет. Выбор по тем же OOF —
    оценка выбора оптимистична."""
    y = np.asarray(y)
    classes = np.array(sorted(set(y)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(model)
    tag = "" if max_iter == 300 else f"_it{max_iter}"
    svc_preds = {s: classes[svc_scores[s].argmax(1)] for s in svc_scores}
    rows = [report_row("t2-svc-none-c2000-C1 (OOF пересчитан)", y, svc_preds, orgs, folds)]
    result = {"model": model, "sigma": sigma, "C_lr": C, "max_iter_lr": max_iter, "headers": {}}
    for header in ("with", "none"):
        X = load_embeddings(cache_dir, EmbeddingSpec(model, max_tokens, header, len(y)))
        lr = {s: oof_logits(X, y, folds[s], C=C, max_iter=max_iter)[0] for s in folds}
        log(f"{header}: LR OOF готов")
        lr_row = report_row(
            f"t2-emb-{slug}-{header}-lr{tag}",
            y,
            {s: classes[lr[s].argmax(1)] for s in lr},
            orgs,
            folds,
        )
        rows.append(lr_row)
        curve = blend_curve(svc_scores, lr, y, classes, orgs, folds)
        curve_name = f"t2_blend_curve_{slug}_{header}{tag}.csv"
        curve.to_csv(out_dir / curve_name, index=False)
        w, info = pick_blend_weight(
            list(zip(curve["w"], curve["main_macro_f1"], curve["control_macro_f1"], strict=True)),
            sigma,
        )
        best = curve[curve["w"] == info["best_w"]].iloc[0]
        rows.append(
            {
                "id": f"t2-blend-svc-C1-{slug}-{header}-w{info['best_w']:.1f}{tag}",
                "main_macro_f1": best["main_macro_f1"],
                "main_f1_rare": best["main_f1_rare"],
                "control_macro_f1": best["control_macro_f1"],
                "control_f1_rare": best["control_f1_rare"],
            }
        )
        result["headers"][header] = {
            "lr": lr_row,
            "chosen_w": w,
            "best_w": info["best_w"],
            "gain": round(info["gain"], 4),
            "improved": info["improved"],
            "curve_csv": curve_name,
        }
        log(f"{header}: лучший w={info['best_w']} прирост {info['gain']:.4f} (σ {sigma})")
    pd.DataFrame(rows).to_csv(out_dir / f"t2_embed_eval_{slug}{tag}.csv", index=False)
    (out_dir / f"t2_embed_eval_{slug}{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=float)
    )
    return result


def svc_oof_paths(out_dir: str | Path, exp_id: str) -> dict[str, Path]:
    return {s: Path(out_dir) / f"oof_{exp_id}_{s}.npy" for s in ("main", "control")}


def compute_svc_oof(cfg: T2Config, exp, out_dir: str | Path, log=print) -> dict[str, np.ndarray]:
    """OOF decision_function TF-IDF + LinearSVC (`exp`) для обеих схем, с кэшем на диск
    (`outputs/oof_<id>_<схема>.npy`; строки в порядке train)."""
    from itgorod.t2_experiment import load_meta, make_schemes

    meta = load_meta(cfg)
    classes = np.array(sorted(set(meta["code"])))
    schemes = make_schemes(meta, cfg.seed)
    paths = svc_oof_paths(out_dir, exp.id)
    out = {}
    for s, folds in schemes.items():
        if paths[s].exists() and np.load(paths[s]).shape == (len(meta), len(classes)):
            out[s] = np.load(paths[s])
            log(f"OOF SVC {s}: из кэша")
            continue
        out[s] = oof_tfidf_scores(cfg, folds, exp, classes)
        np.save(paths[s], out[s])
        log(f"OOF SVC {s}: посчитан")
    return out
