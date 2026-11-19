# %% [markdown]
# # T1 EDA: прогноз `yearly_cv_processed` на 2024 (MAPE)
# Запуск: `uv run python notebooks/01_t1_eda.py`. Графики → `docs/figures/t1_*.png`.
# Правило утечки A: модельные проверки только через временные фолды из `configs/t1.yaml`.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from itgorod import eda
from itgorod.config import REPO_ROOT, load_t1_config
from itgorod.data import read_t1, read_t1_raw

pd.set_option(
    "display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format
)
FIG = REPO_ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
cfg = load_t1_config(REPO_ROOT / "configs" / "t1.yaml")
raw = read_t1_raw(cfg.paths.train)
train = read_t1(cfg.paths.train)
test = read_t1(cfg.paths.test)
print("train", raw.shape, "| train без пустой строки", train.shape, "| test", test.shape)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"t1_{name}.png", dpi=110)
    plt.close()


# %% [markdown]
# ## 1. Качество данных

# %%
q = eda.quality_report(raw)
for k in [
    "n_rows",
    "n_fully_empty",
    "n_no_key_not_empty",
    "n_dup_dept_year",
    "n_inbound_zero",
    "n_turnover_gt1",
    "turnover_gt1_depts",
    "inbound_quantiles",
    "inbound_top",
    "n_rows_any_nan",
]:
    print(k, "=", q[k])
print("нулевые inbound по годам:", q["inbound_zero_by_year"].to_dict())
print("\nпропуски по колонкам:\n", q["nan_by_col"][q["nan_by_col"] > 0].to_string())
print("\nпропуски по колонкам × год:\n", q["nan_by_col_year"].to_string())
same = train["yearly_cv_processed"].isna() & train["active_projects_count"].isna()
print(
    "строк с пропуском И таргета, И проектов:",
    int(same.sum()),
    "| пропусков таргета:",
    int(train["yearly_cv_processed"].isna().sum()),
)
zero = train[train["inbound_applications"] == 0]
pos = train[train["inbound_applications"] > 0]
print(
    "таргет при inbound==0 / >0 (среднее):",
    round(zero["yearly_cv_processed"].mean(), 1),
    round(pos["yearly_cv_processed"].mean(), 1),
    "| медиана inbound/проекты при >0:",
    round((pos["inbound_applications"] / pos["active_projects_count"]).median(), 2),
)
print("нулей inbound в test:", int((test["inbound_applications"] == 0).sum()))
print(
    "turnover>1, отделы/годы:\n",
    train.loc[train["turnover_rate"] > 1, ["department_id", "year", "turnover_rate"]].to_string(
        index=False
    ),
)
print("\nпропуски test:\n", test.isna().sum()[test.isna().sum() > 0].to_string())
print("\nвсе id test есть в train:", bool(test["department_id"].isin(train["department_id"]).all()))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(q["nan_by_col_year"], annot=True, fmt="d", cmap="Reds", ax=ax[0])
ax[0].set_title("Пропуски train: колонка × год")
sns.histplot(train["turnover_rate"].dropna(), bins=60, ax=ax[1])
ax[1].axvline(1, color="r", ls="--")
ax[1].set_title("turnover_rate (красная линия = 1)")
save("quality")

# %% [markdown]
# ## 2. Таргет

# %%
ts = eda.target_summary(train)
print(ts["by_year"].to_string())
print("skew:", round(ts["skew"], 2), "| skew(log):", round(ts["log_skew"], 2))
print("малые значения (<100, <200):", ts["small"], "| минимумы:", ts["min_values"])
print(
    "доля веса 1/y у y<200 (вклад в MAPE при равной отн. ошибке):",
    round(ts["mape_contrib_small"], 3),
)
print("Spearman с таргетом:\n", ts["spearman"].to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(train[eda.TARGET].dropna(), bins=60, ax=ax[0])
ax[0].set_title("yearly_cv_processed")
sns.histplot(np.log(train[eda.TARGET].dropna()), bins=60, ax=ax[1])
ax[1].set_title("log(yearly_cv_processed)")
sns.boxplot(data=train, x="year", y=eda.TARGET, ax=ax[2])
ax[2].set_title("по годам")
save("target")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(
    ax, ["active_projects_count", "current_staff_count", "total_salary_budget"], strict=False
):
    sns.scatterplot(data=train, x=c, y=eda.TARGET, hue="year", s=10, ax=a, palette="viridis")
    a.set_title(f"target vs {c}")
save("target_vs_related")

# %% [markdown]
# ## 3. Гипотеза: target ≈ ratio_отдела × active_projects_count

# %%
rs = eda.ratio_stability(train)
print("CV ratio по всей выборке:", round(rs["overall_cv"], 4))
print("CV ratio внутри отдела:\n", rs["cv_within_dept"].to_string())
print(
    "|ratio/медиана_отдела − 1|, квантили 50/90/99%:",
    rs["row_dev_quantiles"],
    "| строк с отклонением > 15%:",
    rs["n_rows_dev_gt15pct"],
)
print("отделов с CV внутри > 10%:", rs["n_depts_cv_gt10pct"])
print(
    "std средних ratio МЕЖДУ отделами:",
    round(rs["between_dept_std"], 3),
    "| средний std ВНУТРИ отдела:",
    round(rs["within_dept_std_mean"], 3),
)
print("\nratio по годам:\n", rs["by_year"].to_string())
print("\nratio по tech_stack:\n", rs["by_stack"].to_string())
print("\nнаклон ratio по годам внутри отдела:\n", rs["slope_per_year"].to_string())

dev = (d0 := eda.add_ratio(train).dropna(subset=["ratio"]))["ratio"] / d0.groupby("department_id")[
    "ratio"
].transform("median") - 1
bad = d0[dev.abs() > 0.15]
print(
    "строки с отклонением >15%: по годам",
    bad["year"].value_counts().to_dict(),
    "| проекты min/медиана:",
    bad["active_projects_count"].min(),
    bad["active_projects_count"].median(),
)
diag = eda.ratio_model_diagnostics(train, cfg.folds)
print("\nMAPE по временным фолдам (диагностика, не бейзлайн):\n", diag.to_string())

d = eda.add_ratio(train).dropna(subset=["ratio"])
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(d["ratio"], bins=60, ax=ax[0])
ax[0].set_title("ratio = target / projects")
sns.boxplot(data=d, x="tech_stack", y="ratio", ax=ax[1])
ax[1].tick_params(axis="x", rotation=90)
ax[1].set_title("ratio по tech_stack")
sample_ids = d["department_id"].drop_duplicates().sample(40, random_state=cfg.seed)
for _, g in d[d["department_id"].isin(sample_ids)].groupby("department_id"):
    ax[2].plot(g["year"], g["ratio"], alpha=0.5)
ax[2].set_title("ratio по годам, 40 случайных отделов")
save("ratio")

# %% [markdown]
# ## 4. Выражается ли таргет формулой из признаков того же года

# %%
print("точные тождества между признаками (отношение должно быть константой):")
print(eda.identities(train).to_string())
sig = eda.feature_target_signal(train, cfg.folds)
print(
    "\nодиночные признаки: Spearman и MAPE лог-линейной модели (среднее по фолдам):\n",
    sig.to_string(),
)
print("\nКорреляции hired/fired/inbound с ratio:")
print(
    eda.add_ratio(train)[
        [
            "ratio",
            "devs_hired",
            "devs_fired",
            "inbound_applications",
            "turnover_rate",
            "share_seniors",
            "tech_brand_rating",
            "avg_dev_salary",
        ]
    ]
    .corr(method="spearman")["ratio"]
    .round(3)
    .to_string()
)

# %% [markdown]
# ## 5. Сдвиг распределений: train 2023 vs test 2024

# %%
dr = eda.drift_table(train, test, 2023)
print(dr.to_string())

fig, ax = plt.subplots(3, 4, figsize=(18, 10))
for a, c in zip(ax.ravel(), dr.index, strict=False):
    sns.kdeplot(
        train.loc[train["year"] == 2023, c].dropna(), ax=a, label="train 2023", fill=True, alpha=0.3
    )
    sns.kdeplot(test[c].dropna(), ax=a, label="test 2024", fill=True, alpha=0.3)
    a.set_title(f"{c} (KS={dr.loc[c, 'ks']:.2f})")
    a.legend(fontsize=7)
save("drift")

# %% [markdown]
# ## 6. tech_stack

# %%
sc = eda.stack_constancy(train, test)
print({k: v for k, v in sc.items() if k != "depts_per_stack"})
print(sc["depts_per_stack"].to_string())
print("\nсредний таргет по стеку и году:")
print(
    train.pivot_table(
        index="tech_stack", columns="year", values=eda.TARGET, aggfunc="mean"
    ).to_string()
)
