"""T2: схемы валидации (основная stratified, контрольная grouped), срезы OOF, правило A для fit.

Схема закреплена в `docs/DECISIONS.md`, запись 13.
"""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from itgorod.t2_recon import body_hash, parse_header

_PUNCT_RE = re.compile(r"[^\w\s]|_")


def normalize_org(org: str | None) -> str | None:
    """casefold, знаки препинания → пробел, схлопывание пробелов; пусто → None."""
    if org is None or (isinstance(org, float) and np.isnan(org)):
        return None
    s = " ".join(_PUNCT_RE.sub(" ", str(org).casefold()).split())
    return s or None


def build_meta(texts: Sequence[str], hash_chars: int = 2000) -> pd.DataFrame:
    """Для каждого документа: нормализованный орган и хеш тела (без шапки). Тексты
    нужны только в усечённом виде (≥ шапка + `hash_chars` тела)."""
    rows = [
        {"org_norm": normalize_org(parse_header(t)["org"]), "body_hash": body_hash(t, hash_chars)}
        for t in texts
    ]
    return pd.DataFrame(rows, columns=["org_norm", "body_hash"])


def group_ids(body_hashes: Sequence, orgs: Sequence) -> np.ndarray:
    """Компоненты связности: строки с общим телом или общим органом — одна группа.
    Строка без органа не связывается органом ни с кем."""
    n = len(body_hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_by(keys: Sequence) -> None:
        first: dict = {}
        for i, k in enumerate(keys):
            if k is None or (isinstance(k, float) and np.isnan(k)):
                continue
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i

    union_by(body_hashes)
    union_by(orgs)
    roots = np.array([find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def main_folds(codes: Sequence, body_hashes: Sequence, n_splits: int, seed: int) -> np.ndarray:
    """`StratifiedKFold` по `code`; точные дубли тела всегда в одном фолде: разбиваются
    представители групп (метка представителя — первая в группе), члены идут за ним."""
    codes = np.asarray(codes, dtype=object)
    uniq, first_idx, inverse = np.unique(
        np.asarray(body_hashes, dtype=object).astype(str), return_index=True, return_inverse=True
    )
    rep_y = codes[first_idx]
    fold_rep = np.empty(len(uniq), dtype=int)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(uniq)), rep_y)):
        fold_rep[va] = k
    return fold_rep[inverse]


def control_folds(
    codes: Sequence, body_hashes: Sequence, orgs: Sequence, n_splits: int, seed: int
) -> np.ndarray:
    """`StratifiedGroupKFold` по группам «орган ∪ тело»: орган не пересекает train и valid."""
    groups = group_ids(body_hashes, orgs)
    folds = np.empty(len(groups), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(np.zeros(len(groups)), np.asarray(codes), groups)):
        folds[va] = k
    return folds


def seen_org_mask(orgs: pd.Series, folds: np.ndarray) -> np.ndarray:
    """Для каждой строки: орган есть среди органов train-части её фолда (None → False)."""
    orgs = pd.Series(orgs).reset_index(drop=True)
    out = np.zeros(len(orgs), dtype=bool)
    for k in np.unique(folds):
        va = folds == k
        train_orgs = set(orgs[~va].dropna())
        out[va] = orgs[va].isin(train_orgs).to_numpy()
    return out


def fit_fold_features(vec, train_texts: Sequence[str], valid_texts: Sequence[str]):
    """Правило A: `fit` векторайзера — только на train-части, valid лишь `transform`."""
    return vec.fit_transform(list(train_texts)), vec.transform(list(valid_texts))


def oof_report(
    y_true: Sequence, y_pred: Sequence, seen: np.ndarray, rare_threshold: int = 50
) -> dict:
    """Macro-F1 на всём OOF, на строках с известным/неизвестным органом, по классам и на
    редких классах (<`rare_threshold` примеров). Срезы считаются по классам, присутствующим
    в истинных метках среза."""
    y_true, y_pred, seen = np.asarray(y_true), np.asarray(y_pred), np.asarray(seen, dtype=bool)
    classes = sorted(set(y_true))
    support = pd.Series(y_true).value_counts()

    def macro(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        labels = sorted(set(y_true[mask]))
        return float(
            f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0)
        )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = pd.DataFrame({"code": classes, "n": s, "precision": p, "recall": r, "f1": f})
    rare = sorted(support[support < rare_threshold].index)
    return {
        "macro_f1": macro(np.ones(len(y_true), dtype=bool)),
        "macro_f1_seen": macro(seen),
        "macro_f1_unseen": macro(~seen),
        "n_seen": int(seen.sum()),
        "n_unseen": int((~seen).sum()),
        "rare_classes": rare,
        "macro_f1_rare": float(per_class.loc[per_class["code"].isin(rare), "f1"].mean())
        if rare
        else float("nan"),
        "per_class": per_class,
    }
