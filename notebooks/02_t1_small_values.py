# %% [markdown]
# # T1: малые значения таргета (y < 200) на фолдах 2022 / 2023
# Запуск: `uv run python notebooks/02_t1_small_values.py`. Правило A: все модели и поправки —
# только по train-годам фолда; таргет valid-года используется лишь для оценки.
# Гипотезы: (1) y = a + b·projects с intercept; (2) сдвиг ratio у отдела; (3) дискретность проектов.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itgorod import eda
from itgorod.config import REPO_ROOT, load_t1_config
from itgorod.data import read_t1
from itgorod.t1 import BASELINE
from itgorod.t1_experiment import compare
from itgorod.t1_experiment import fold_predictions as runner_predictions

pd.set_option(
    "display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format
)
cfg = load_t1_config(REPO_ROOT / "configs" / "t1.yaml")
exp = cfg.experiment
train = read_t1(cfg.paths.train)
SMALL = exp.small_threshold
COLS = [
    "department_id",
    "y",
    "p_year",
    "p_proj",
    "proj",
    "p_ratio",
    "ratio",
    "staff",
    "pred",
    "ape",
]

# %% [markdown]
# ## 0. Строки y < 200: проекты Y-1 и Y, ratio Y-1 и фактический, сотрудники, ошибка

# %%
rows = {f.valid_year: eda.last_year_rows(train, f) for f in cfg.folds}
for f in cfg.folds:
    r, s = rows[f.valid_year], eda.small_rows(train, f, SMALL)
    print(
        f"\n=== фолд {f.valid_year}: строк {len(r)}, малых {len(s)}; "
        f"MAPE всего {r['ape'].mean():.4f}, "
        f"малых {s['ape'].mean():.4f}, остальных {r.loc[r['y'] >= SMALL, 'ape'].mean():.4f}; "
        f"доля малых в сумме APE {s['ape'].sum() / r['ape'].sum():.3f}"
    )
    print(s[COLS].to_string(index=False))
    if len(s) > 1:
        worst = s.loc[s["ape"].idxmax(), "department_id"]
        rest = s[s["department_id"] != worst]
        print(f"без худшей ({int(worst)}): MAPE малых {rest['ape'].mean():.4f}, n = {len(rest)}")
    print("строк, где год последних данных не Y-1:", int((r["p_year"] != f.valid_year - 1).sum()))

# %% [markdown]
# ## Гипотеза 1: y = a + b·projects с ненулевым intercept (⇒ эластичность α < 1 у малых отделов)
# Группы — по среднему числу проектов отдела за train-годы фолда.

# %%
BINS = [0, 10, 20, 50, 1e9]
for f in cfg.folds:
    print(f"\nα по размеру отдела, годы <= {f.train_max_year} (train фолда {f.valid_year}):")
    print(
        eda.elasticity_by_size(train, f.train_max_year, BINS, exp.n_bootstrap, cfg.seed).to_string(
            index=False
        )
    )
ic = eda.dept_intercepts(train)
ic["group"] = eda._group_labels(ic["mean_proj"], BINS)
print("\nпо-отдельный intercept (все годы, описательно): a/mean(y), t = a/se(a)")
print(
    ic.groupby("group", observed=True)
    .agg(
        n=("t", "size"),
        a_rel_median=("a_rel", "median"),
        t_median=("t", "median"),
        share_abs_t_gt2=("t", lambda s: (s.abs() > 2).mean()),
    )
    .to_string()
)

# %% [markdown]
# ## Гипотеза 2: сдвиг уровня ratio
# Сначала: насколько |Δlog ratio| малых строк необычен среди всех строк valid-года.

# %%
for f in cfg.folds:
    r = rows[f.valid_year]
    s = r[r["y"] < SMALL]
    pct = [(r["dlr"].abs() < v).mean() for v in s["dlr"].abs()]
    print(f"\nфолд {f.valid_year}: |Δlog ratio| малых строк и процентиль среди строк valid-года")
    print(
        pd.DataFrame(
            {"department_id": s["department_id"], "abs_dlr": s["dlr"].abs(), "pctl": pct}
        ).to_string(index=False)
    )
    print(
        f"медиана |Δlog ratio| по всем строкам: {r['dlr'].abs().median():.4f}; "
        f"доля строк с > 0,15: "
        f"{(r['dlr'].abs() > 0.15).mean():.3f}"
    )

# %% [markdown]
# Где сдвиг ratio реально большой: строки с резким изменением проектов |Δlog projects| > 0,4.

# %%
JUMP = 0.4
for f in cfg.folds:
    r = rows[f.valid_year]
    j = r[r["dlp"].abs() > JUMP]
    print(
        f"\nфолд {f.valid_year}: строк с |Δlog проектов| > {JUMP}: {len(j)} "
        f"(малых y среди них: {int((j['y'] < SMALL).sum())})"
    )
    print(
        j[
            ["department_id", "y", "p_proj", "proj", "p_ratio", "ratio", "dlp", "dly", "ape"]
        ].to_string(index=False)
    )
    rest = r[r["dlp"].abs() <= JUMP]
    print(
        f"MAPE на строках с резким изменением проектов {j['ape'].mean():.4f} (n={len(j)}), "
        f"на остальных {rest['ape'].mean():.4f}"
    )
    big = r[r["ape"] > 0.2]
    print(
        f"строк с APE > 20%: {len(big)}, из них резкое изменение проектов: "
        f"{int((big['dlp'].abs() > JUMP).sum())}, "
        f"малый y: {int((big['y'] < SMALL).sum())}"
    )

# обвалы проектов (dlp < -0.4) за все годы: есть ли они в train-годах фолдов (правило A)?
d = train.sort_values(["department_id", "year"]).copy()
for col, new in (("active_projects_count", "pp"), (eda.TARGET, "py"), ("year", "pyr")):
    d[new] = d.groupby("department_id")[col].shift()
d = d[(d["year"] - d["pyr"] == 1)].dropna(subset=[eda.TARGET, "py", "active_projects_count", "pp"])
d["dlp"], d["dly"] = np.log(d["active_projects_count"] / d["pp"]), np.log(d[eda.TARGET] / d["py"])
print("\nпереходы год→год с обвалом проектов (Δlog < -0,4), все годы:")
print(
    d[d["dlp"] < -JUMP][
        ["department_id", "year", "pp", "active_projects_count", "py", eda.TARGET, "dlp", "dly"]
    ].to_string(index=False)
)
for t in (2021, 2022, 2023):
    x = d[d["year"] == t]
    print(
        f"  переход в {t}: {len(x)} строк, обвалов (< -0,4): {int((x['dlp'] < -JUMP).sum())}, "
        f"умеренных падений (-0,4..-0,2): {int(((x['dlp'] >= -JUMP) & (x['dlp'] < -0.2)).sum())}"
    )

# %% [markdown]
# ## Гипотеза 3: дискретность проектов (±1 проект = большой относительный скачок)
# Шум ratio по размеру отдела (по проектам в предыдущем году) и MAPE планки vs среднего ratio.

# %%
d3 = train.sort_values(["department_id", "year"]).copy()
d3["r"] = d3[eda.TARGET] / d3["active_projects_count"]
d3["dlr"] = d3.groupby("department_id")["r"].transform(lambda s: np.log(s).diff())
d3["pp"] = d3.groupby("department_id")["active_projects_count"].shift()
d3["grp"] = eda._group_labels(d3["pp"], [0, 5, 10, 20, 50, 1e9])
print("|Δlog ratio| год к году по числу проектов в прошлом году (все годы, описательно):")
print(
    d3.groupby("grp", observed=True)["dlr"]
    .agg(n="count", median=lambda s: s.abs().median(), mean=lambda s: s.abs().mean())
    .to_string()
)

for f in cfg.folds:
    y, preds = eda.fold_predictions(train, f)
    r = rows[f.valid_year].set_index("department_id")
    t = pd.DataFrame(
        {"grp": eda._group_labels(r["p_proj"], [0, 20, 50, 1e9]), "y": y.reindex(r.index)}
    )
    for name in ("ratio_dept_last", "ratio_dept_mean"):
        t[name] = (preds[name].reindex(r.index) - t["y"]).abs() / t["y"]
    print(f"\nфолд {f.valid_year}: MAPE по проектам в Y-1: последний ratio vs средний (train-годы)")
    print(
        t.groupby("grp", observed=True)
        .agg(
            n=("y", "size"),
            n_small_y=("y", lambda v: int((v < SMALL).sum())),
            last=("ratio_dept_last", "mean"),
            mean=("ratio_dept_mean", "mean"),
        )
        .to_string()
    )

# дискретность напрямую: отделы с <= 10 проектов в Y-1; ошибка при изменении проектов и без него
for f in cfg.folds:
    r = rows[f.valid_year]
    t = r[r["p_proj"] <= 10].assign(dproj=lambda x: x["proj"] - x["p_proj"])
    print(
        f"\nфолд {f.valid_year}: отделы с <= 10 проектов в Y-1 (n={len(t)}); "
        "MAPE планки по Δ проектов"
    )
    print(
        t.groupby(t["dproj"].clip(-2, 2).map({-2: "<=-2", -1: "-1", 0: "0", 1: "+1", 2: ">=+2"}))[
            "ape"
        ]
        .agg(n="size", mape="mean", max="max")
        .to_string()
    )

# сколько ошибки дают обвалы проектов, и MAPE без них
for f in cfg.folds:
    r = rows[f.valid_year]
    c = r["dlp"] < -JUMP
    print(
        f"\nфолд {f.valid_year}: обвалов проектов (Δlog < -0,4): {int(c.sum())}; "
        f"их вклад в сумму APE "
        f"{r.loc[c, 'ape'].sum() / r['ape'].sum():.3f}; MAPE всего {r['ape'].mean():.4f} → без них "
        f"{r.loc[~c, 'ape'].mean():.4f}"
    )

# риск для 2024: сколько отделов test резко меняют проекты относительно 2023
# (признаки года Y доступны по правилу A)
test = read_t1(cfg.paths.test)
last = train[train["year"] == 2023].set_index("department_id")["active_projects_count"]
dlp_test = np.log(
    test.set_index("department_id")["active_projects_count"]
    / last.reindex(test["department_id"]).to_numpy()
)
print(
    f"\nтест 2024 vs 2023: |Δlog проектов| > {JUMP}: {int((dlp_test.abs() > JUMP).sum())} "
    f"(обвалов < -{JUMP}: {int((dlp_test < -JUMP).sum())}, "
    f"ростов: {int((dlp_test > JUMP).sum())}); "
    f"для сравнения valid 2022 / 2023: "
    f"{int((rows[2022]['dlp'].abs() > JUMP).sum())} / {int((rows[2023]['dlp'].abs() > JUMP).sum())}"
)

# %% [markdown]
# ## Кандидат-поправка из гипотезы 1: y_last · (proj / p_proj)^α (α — по train-годам фолда)
# Сравнение с планкой тем же раннером: общие строки, построчное сравнение, парный бутстрэп.

# %%
E2_BINS = [0, 50, 1e9]  # заданы заранее: в группе до 20 проектов всего 16 отделов
specs = {
    "baseline": (BASELINE, False),
    "E": lambda h, r: eda.predict_elastic(h, r),  # одна α на всех
    "E2": lambda h, r: eda.predict_elastic(h, r, bins=E2_BINS),  # α по группам, до просмотра
}
res = []
for f in cfg.folds:
    fp = runner_predictions(train, f, specs)
    hist = train[train["year"] <= f.train_max_year]
    alphas = {"E": eda.elasticity_alpha(hist)}
    e2 = eda.elasticity_by_size(train, f.train_max_year, E2_BINS, 0, cfg.seed)
    print(
        f"фолд {f.valid_year}: α (E) = {alphas['E']:.3f}; α по группам E2:",
        dict(zip(e2["group"], e2["alpha"].round(3), strict=True)),
    )
    for model in ("E", "E2"):
        for label, mask in (
            ("все", fp["y"] > 0),
            (f"y<{SMALL:g}", fp["y"] < SMALL),
            (f"y>={SMALL:g}", fp["y"] >= SMALL),
        ):
            sub = fp[mask]
            m = compare(
                sub["y"],
                sub[f"pred_{model}"],
                sub["pred_baseline"],
                SMALL,
                exp.n_bootstrap,
                cfg.seed,
            )
            base = float(np.mean(np.abs(sub["pred_baseline"] - sub["y"]) / sub["y"]))
            res.append(
                {
                    "фолд": f.valid_year,
                    "модель": model,
                    "срез": label,
                    "n": m["n"],
                    "MAPE_base": base,
                    "MAPE_cand": m["mape"],
                    "Δ": m["delta_mape"],
                    "Δ_lo": m["delta_lo"],
                    "Δ_hi": m["delta_hi"],
                    "улучшено": m["share_improved"],
                }
            )
print(pd.DataFrame(res).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# %% [markdown]
# ## График: Δlog y против Δlog projects (valid-строки обоих фолдов)

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for a, f in zip(ax, cfg.folds, strict=True):
    r = rows[f.valid_year]
    small = r["y"] < SMALL
    a.scatter(r.loc[~small, "dlp"], r.loc[~small, "dly"], s=8, alpha=0.4, label=f"y>={SMALL:g}")
    a.scatter(r.loc[small, "dlp"], r.loc[small, "dly"], s=40, c="r", label=f"y<{SMALL:g}")
    lim = np.array([r["dlp"].min(), r["dlp"].max()])
    a.plot(lim, lim, "k--", lw=1, label="y ∝ projects (α = 1)")
    a.set_title(f"фолд {f.valid_year}")
    a.set_xlabel("Δlog проектов (Y vs Y-1)")
    a.legend(fontsize=8)
ax[0].set_ylabel("Δlog y")
plt.tight_layout()
plt.savefig(REPO_ROOT / "docs" / "figures" / "t1_small_values.png", dpi=110)
