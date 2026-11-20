# %% [markdown]
# # T2: зонд «что в шапке предсказывает класс»
# Запуск: `uv run python notebooks/04_t2_header_probe.py`. Для каждого признака шапки (дата
# выгрузки, дата акта, месяц акта, формат номера, тип акта) класс предсказывается модой класса
# по значению признака; мода считается только по train-части фолда (правило A), оценка — на
# valid-части обеих схем (main, control). Это зонд, не модель; результаты — записи 17.

# %%
import dataclasses

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from itgorod.config import REPO_ROOT, load_t2_config
from itgorod.data import build_t2_cache
from itgorod.t2_experiment import load_meta, load_train, make_schemes
from itgorod.t2_recon import split_header

cfg = load_t2_config(REPO_ROOT / "configs/t2.yaml")
meta = load_meta(cfg)
head = load_train(cfg, 2000)["text"].to_numpy()
h = pd.DataFrame([split_header(t) for t in head])
h["act_month"] = h.act_date.str[3:]
h["num_len"] = h.number.fillna("").str.replace(r"\d", "9", regex=True)
y = meta.code.to_numpy()
folds = make_schemes(meta, cfg.seed)
print(
    "n unique fetch dates",
    h.fetch_date.nunique(),
    "act dates",
    h.act_date.nunique(),
    "numbers",
    h.number.nunique(),
)
for feat in ["fetch_date", "act_date", "act_month", "num_len", "type"]:
    for sch in ("main", "control"):
        f = folds[sch]
        pred = np.empty(len(y), dtype=object)
        for k in range(5):
            tr = f != k
            va = f == k
            mode = (
                pd.DataFrame({"x": h[feat][tr].fillna("NA").to_numpy(), "y": y[tr]})
                .groupby("x")
                .y.agg(lambda s: s.value_counts().index[0])
            )
            glob = pd.Series(y[tr]).value_counts().index[0]
            pred[va] = h[feat][va].fillna("NA").map(mode).fillna(glob).to_numpy()
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        print(f"{feat:11s} {sch:8s} acc {np.mean(pred == y):.3f} macroF1 {f1:.3f}")
print("majority acc", pd.Series(y).value_counts(normalize=True).iloc[0].round(3))


# %% [markdown]
# ## Распределение даты выгрузки: train против test
# Если доли по датам совпадают, зависимость класса от даты выгрузки (если она связана с порядком
# сбора) скорее всего присутствует и в test; без меток test это проверить нельзя.

# %%
test_texts = pd.read_parquet(
    build_t2_cache(dataclasses.replace(cfg, max_chars=2000), "test", smoke=False)
)["text"]
fetch = lambda texts: pd.Series([split_header(t)["fetch_date"] for t in texts]).fillna("NA")  # noqa: E731
share = pd.concat(
    [fetch(head).value_counts(normalize=True), fetch(test_texts).value_counts(normalize=True)],
    axis=1,
    keys=["train", "test"],
).fillna(0)
print((share.T * 100).round(1).to_string())
print(
    "макс. расхождение доли по дате, п.п.:",
    round(((share.train - share.test).abs() * 100).max(), 2),
)
