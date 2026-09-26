"""Собирает solution.ipynb из ячеек ниже (единый источник кода финального решения)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Предсказание эскалации AML-алертов — воспроизводимое решение

**Задача:** для каждого `signal_id` из `test_signals.csv` предсказать вероятность эскалации. Метрика — ROC-AUC.

**Ключевая идея (из EDA):** у каждого клиента есть скрытый класс **A/B**. У каждого класса в каждом сегменте
(тип транзакции × направление) свой **жёсткий минимум суммы**; значения ниже минимума «прижаты» в узкую полосу
над ним. Поэтому транзакции у порога выдают класс клиента (6 045 клиентов определяются правилом без единого
противоречия). Эскалация: класс A ≈ 10%, класс B ≈ 30%. Для остальных клиентов класс восстанавливается
классификатором по «форме» профиля сумм (обучен на якорях из полос банковских переводов, **без таргета**).

**Пайплайн:**
1. Признаки профиля клиента по 8 сегментам (n, доли, mean/std/квантили/min/max, относительные уровни),
   временные окна, потоки, гистограммы сумм.
2. Признаки скрытого класса: полосы у порога + вероятность класса B (`lat_p`).
3. LightGBM + CatBoost + XGBoost на одинаковых repeated stratified 5-fold × 3 фолдах, 3 seed'а на фолд.
4. Бленд — среднее рангов; тестовый прогноз — среднее 135 фолдовых моделей.

**CV ROC-AUC (OOF, 3 повтора):** LightGBM 0.6641 · CatBoost 0.6642 · XGBoost 0.6631 · **бленд 0.6643**.

**Запуск:** положить данные в `fintech_data/` рядом с ноутбуком, `pip install -r requirements.txt`,
*Restart & Run All*. Время ≈ 3 минуты на 8 ядрах (Apple Silicon, 16 ГБ RAM). Результат — `team_<TEAM_ID>.csv`.
""")

code(r"""
TEAM_ID = "F2F427DD"

import os, random, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
N_JOBS = 8
DATA = Path("fintech_data")
TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
SEGS = [f"{t}_{d}" for t in TYPES for d in ("in", "out")]   # seg = type*2 + (chiqim)
BURST_SEC = 600      # транзакции за последние 10 минут до signal_sanasi ("всплеск")
DAY = 86400.0
T0 = time.time()
print({m.__name__: m.__version__ for m in (np, pd, lgb, xgb)})
""")

md("## 1. Данные")
code(r"""
tr = pd.read_csv(DATA / "train_signals.csv", parse_dates=["signal_sanasi"]); tr["is_test"] = 0
te = pd.read_csv(DATA / "test_signals.csv", parse_dates=["signal_sanasi"]); te["is_test"] = 1; te["eskalatsiya"] = np.nan
S = pd.concat([tr, te], ignore_index=True).sort_values("signal_id").reset_index(drop=True)

tx = pd.concat([pd.read_parquet(DATA / "train_transactions.parquet"),
                pd.read_parquet(DATA / "test_transactions.parquet")], ignore_index=True)
tx = tx.merge(S[["signal_id", "signal_sanasi"]], on="signal_id", how="left")
tx["sec"] = (tx.signal_sanasi - tx.tranzaksiya_vaqti).dt.total_seconds().astype("float64")   # секунд до сигнала
tx["a"] = tx.miqdor_indeksi.astype("float64")
tx["out"] = (tx.kirim_chiqim == "chiqim").astype("int8")
tx["typ"] = tx.tranzaksiya_turi.map({t: i for i, t in enumerate(TYPES)}).astype("int8")
tx["seg"] = (tx.typ * 2 + tx.out).astype("int8")
tx["burst"] = (tx.sec <= BURST_SEC).astype("int8")
tx["hour"] = tx.tranzaksiya_vaqti.dt.hour.astype("int8")
tx["dow"] = tx.tranzaksiya_vaqti.dt.dayofweek.astype("int8")
tx = tx.sort_values(["signal_id", "tranzaksiya_vaqti", "a"], kind="mergesort").reset_index(drop=True)
tx = tx[["signal_id", "tranzaksiya_vaqti", "sec", "a", "out", "typ", "seg", "burst", "hour", "dow"]]
print(S.shape, tx.shape, "positives:", int(tr.eskalatsiya.sum()), f"rate={tr.eskalatsiya.mean():.4f}")
""")

md(r"""
## 2. Проверка главной находки: классовые минимумы сумм
Транзакция в зоне `[минимум сегмента, минимум другого класса)` однозначно определяет класс клиента.
Ниже — число клиентов с уликами каждого класса и число противоречий (ожидается 0).
""")
code(r"""
# минимум класса (консервативно) как смещение от глобального минимума сегмента: seg -> (A_min, B_min)
CLASS_FLOOR = {0: (0.03, 0.0), 1: (0.14, 0.0), 2: (0.0, 0.015), 3: (0.0, 0.03),
               4: (0.0, 0.04), 5: (0.0, 0.12), 6: (0.0, 0.06), 7: (0.45, 0.0)}
floor = tx.groupby("seg").a.min()
off = tx.a - tx.seg.map(floor)
evA = pd.Series(0, index=S.signal_id); evB = pd.Series(0, index=S.signal_id)
for sg, (fa, fb) in CLASS_FLOOR.items():
    z = (tx.seg == sg) & (off < max(fa, fb))
    c = z.groupby(tx.signal_id).sum().reindex(S.signal_id).fillna(0)
    if fa > fb: evB += c
    else: evA += c
det = pd.Series(np.nan, index=S.signal_id); det[(evB > 0) & (evA == 0)] = 1; det[(evA > 0) & (evB == 0)] = 0
print("клиентов с уликой B:", int((evB > 0).sum()), "| A:", int((evA > 0).sum()),
      "| противоречий:", int(((evA > 0) & (evB > 0)).sum()))
ys = S.set_index("signal_id").eskalatsiya
print("доля эскалаций по классу (train):", ys.groupby(det).mean().round(3).to_dict(),
      "| не определён:", round(ys[det.isna()].mean(), 3))
""")

md("## 3. Признаки сигнала")
code(r"""
def g_profile(tx):
    # Параметры 'профиля клиента' по 8 сегментам + уровни относительно самого клиента.
    g = tx.groupby(["signal_id", "seg"]).a
    st = pd.DataFrame({"n": g.size(), "mean": g.mean(), "std": g.std(), "med": g.median(),
                       "q10": g.quantile(.1), "q90": g.quantile(.9), "min": g.min(), "max": g.max()}).unstack()
    st.columns = [f"{s}_{SEGS[c]}" for s, c in st.columns]
    n = tx.groupby("signal_id").size()
    F = pd.DataFrame(index=n.index)
    F["n"] = n
    for sg in SEGS:
        F[f"n_{sg}"] = st[f"n_{sg}"].reindex(F.index).fillna(0)
        F[f"frac_{sg}"] = F[f"n_{sg}"] / n
    for k in ["mean", "std", "med", "q10", "q90", "min", "max"]:
        for sg in SEGS:
            F[f"{k}_{sg}"] = st[f"{k}_{sg}"]
    gm = tx.groupby("seg").a.mean()
    dev = tx.a - tx.seg.map(gm)
    F["lvl_all"] = dev.groupby(tx.signal_id).mean()
    nonbank = tx.typ != 1
    F["lvl_nonbank"] = dev[nonbank].groupby(tx.signal_id[nonbank]).mean()
    F["lvl_card"] = dev[tx.typ == 0].groupby(tx.signal_id[tx.typ == 0]).mean()
    for ref in ["lvl_nonbank", "lvl_card"]:
        for sg_i, sg in enumerate(SEGS):
            F[f"rel_{ref}_{sg}"] = F[f"mean_{sg}"] - gm[sg_i] - F[ref]
    base = ["karta_in", "karta_out", "bank_otkazmasi_in", "bank_otkazmasi_out", "naqd_in", "naqd_out"]
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            F[f"d_{a}__{b}"] = F[f"mean_{a}"] - F[f"mean_{b}"]
    F["bank_minus_card"] = (F.mean_bank_otkazmasi_in.fillna(F.mean_bank_otkazmasi_out) + F.mean_bank_otkazmasi_out.fillna(F.mean_bank_otkazmasi_in)) / 2 \
        - (F.mean_karta_in.fillna(F.mean_karta_out) + F.mean_karta_out.fillna(F.mean_karta_in)) / 2
    for sg in SEGS[1:]:
        F[f"stdr_{sg}"] = F[f"std_{sg}"] / F["std_karta_in"]
    F["a_mean"] = tx.groupby("signal_id").a.mean()
    F["a_std"] = tx.groupby("signal_id").a.std()
    return F


def g_time(tx, s):
    # Активность по окнам до сигнала, всплеск, интервалы, час/день недели.
    F = pd.DataFrame(index=s.signal_id)
    sec = tx.sec
    wins = [("burst", -1e12, BURST_SEC), ("w1", BURST_SEC, DAY), ("w7", DAY, 7 * DAY), ("w30", 7 * DAY, 30 * DAY),
            ("w60", 30 * DAY, 60 * DAY), ("w90", 60 * DAY, 90 * DAY), ("w120", 90 * DAY, 120 * DAY),
            ("w150", 120 * DAY, 150 * DAY), ("w180", 150 * DAY, 1e12)]
    for nm, lo, hi in wins:
        m = (sec > lo) & (sec <= hi)
        F[f"cnt_{nm}"] = m.groupby(tx.signal_id).sum()
    n = F.filter(like="cnt_").sum(1)
    for nm, *_ in wins:
        F[f"cf_{nm}"] = F[f"cnt_{nm}"] / n
    b = np.clip((sec / (30 * DAY)).astype(int), 0, 5)
    c = pd.crosstab(tx.signal_id, b).reindex(F.index).fillna(0)
    lc = np.log1p(c.values); x = np.arange(6) - 2.5
    F["act_slope"] = (lc * x).sum(1) / (x ** 2).sum()
    F["first_day"] = tx.groupby("signal_id").sec.max() / DAY
    F["last_sec_nb"] = tx[tx.burst == 0].groupby("signal_id").sec.min()
    F["n_after_sig"] = (sec < 0).groupby(tx.signal_id).sum()
    bt = tx[tx.burst == 1]
    gb = bt.groupby("signal_id")
    F["burst_out"] = gb.out.mean(); F["burst_amean"] = gb.a.mean(); F["burst_astd"] = gb.a.std()
    for i, t in enumerate(TYPES):
        F[f"burst_frac_{t}"] = (bt.typ == i).groupby(bt.signal_id).mean()
    F["burst_span"] = gb.sec.max() - gb.sec.min()
    nb = tx[tx.burst == 0]
    gap = nb.groupby("signal_id").sec.diff(-1).abs()
    gg = gap.groupby(nb.signal_id)
    F["gap_mean"] = gg.mean(); F["gap_med"] = gg.median(); F["gap_std"] = gg.std(); F["gap_max"] = gg.max()
    F["gap_cv"] = F.gap_std / F.gap_mean
    F["gap_lt60"] = (gap < 60).groupby(nb.signal_id).mean(); F["gap_lt3600"] = (gap < 3600).groupby(nb.signal_id).mean()
    hr = nb.hour
    F["night"] = (hr < 6).groupby(nb.signal_id).mean(); F["evening"] = (hr >= 18).groupby(nb.signal_id).mean()
    F["wkend"] = (nb.dow >= 5).groupby(nb.signal_id).mean()
    day = (nb.sec // DAY)
    dc = nb.groupby([nb.signal_id, day]).size()
    F["act_days"] = dc.groupby(level=0).size(); F["d_max"] = dc.groupby(level=0).max(); F["d_std"] = dc.groupby(level=0).std()
    F["d_mean"] = dc.groupby(level=0).mean()
    F["fano"] = F.d_std ** 2 / F.d_mean
    sd = s.set_index("signal_id").signal_sanasi
    F["sig_month"] = sd.dt.month; F["sig_dow"] = sd.dt.dayofweek; F["sig_t"] = (sd - pd.Timestamp("2025-01-01")).dt.days
    return F


def g_flow(tx):
    # Денежные потоки: net-flow, баланс-прокси, pass-through.
    F = pd.DataFrame(index=tx.signal_id.unique())
    v = np.exp(tx.a)
    sgn = np.where(tx.out == 1, -1.0, 1.0)
    F["net_idx"] = (tx.a * sgn).groupby(tx.signal_id).sum()
    F["in_vol"] = (v * (1 - tx.out)).groupby(tx.signal_id).sum()
    F["out_vol"] = (v * tx.out).groupby(tx.signal_id).sum()
    F["out_in_ratio"] = F.out_vol / F.in_vol
    bal = (v * sgn).groupby(tx.signal_id).cumsum()
    gb = bal.groupby(tx.signal_id)
    F["bal_min"] = gb.min() / F.in_vol; F["bal_max"] = gb.max() / F.in_vol
    t = tx[["signal_id", "sec", "out", "a", "typ"]].copy()
    t["t_in"] = np.where(t.out == 0, -t.sec, np.nan)
    t["t_in"] = t.groupby("signal_id").t_in.ffill()
    t["a_in"] = np.where(t.out == 0, t.a, np.nan); t["a_in"] = t.groupby("signal_id").a_in.ffill()
    o = t[t.out == 1]
    dt = (-o.sec) - o.t_in
    F["pt_1h"] = (dt < 3600).groupby(o.signal_id).mean()
    F["pt_1d"] = (dt < DAY).groupby(o.signal_id).mean()
    F["pt_dt_med"] = dt.groupby(o.signal_id).median()
    F["pt_amt_diff"] = (o.a - o.a_in).abs().groupby(o.signal_id).median()
    F["pt_close_1d"] = ((dt < DAY) & ((o.a - o.a_in).abs() < 0.1)).groupby(o.signal_id).mean()
    t["t_bin"] = np.where((t.out == 0) & (t.typ == 1), -t.sec, np.nan); t["t_bin"] = t.groupby("signal_id").t_bin.ffill()
    co = t[(t.out == 1) & (t.typ == 2)]
    F["cash_after_bank_1d"] = ((-co.sec - co.t_bin) < DAY).groupby(co.signal_id).mean()
    return F


def g_hist(tx, nb=12):
    # Гистограмма сумм внутри сегмента по глобальным квантилям сегмента (доли).
    parts = []
    for sg_i in [0, 1, 2, 3, 4, 5]:
        t = tx[tx.seg == sg_i]
        edges = np.quantile(t.a, np.linspace(0, 1, nb + 1)[1:-1])
        b = np.searchsorted(edges, t.a.values)
        c = pd.crosstab(t.signal_id, b)
        c = c.div(c.sum(1), axis=0)
        c.columns = [f"h_{SEGS[sg_i]}_{i}" for i in c.columns]
        parts.append(c)
    return pd.concat(parts, axis=1)


F = pd.concat([g_profile(tx), g_time(tx, S), g_flow(tx), g_hist(tx)], axis=1).reindex(S.signal_id)
F = F.replace([np.inf, -np.inf], np.nan)
print("features:", F.shape, f"{time.time() - T0:.0f}s")
""")

md(r"""
## 4. Скрытый класс клиента (без таргета)
Полосы у порога: значения ниже минимума класса «прижаты» в узкую полосу над ним, у классов A и B полосы разные.
Якорь = полоса банковских переводов (bank_in/bank_out). Классификатор якоря обучается по профилю клиента
**без** признаков, прямо раскрывающих банковскую полосу; OOF-прогноз для якорных клиентов, среднее фолдовых
моделей для остальных. Используются только транзакции (train+test), метка `eskalatsiya` не участвует.
""")
code(r"""
BANDS = {0: [(0, 0.035), (0.165, 0.21)], 1: [(0, 0.01), (0.01, 0.14)],
         2: [(0, 0.015), (0.015, 0.04)], 3: [(0, 0.03), (0.03, 0.07)],
         4: [(0, 0.04)], 5: [(0, 0.04)]}

def band_features(tx):
    fl = tx.groupby("seg").a.min()
    off = tx.a - tx.seg.map(fl)
    out = {}
    for sg, bl in BANDS.items():
        m_sg = tx.seg == sg
        n = m_sg.groupby(tx.signal_id).sum()
        for j, (lo, hi) in enumerate(bl):
            c = (m_sg & (off >= lo) & (off < hi)).groupby(tx.signal_id).sum()
            out[f"band{j}_{SEGS[sg]}_n"] = c
            out[f"band{j}_{SEGS[sg]}_f"] = c / n.replace(0, np.nan)
    return pd.DataFrame(out)

def anchor(B):
    def grp(seg):
        a = B[f"band0_{seg}_n"]; b = B[f"band1_{seg}_n"]
        g = pd.Series(np.nan, index=B.index)
        g[(a > 0) & (b == 0)] = 0; g[(b > 0) & (a == 0)] = 1
        return g
    gi, go = grp("bank_otkazmasi_in"), grp("bank_otkazmasi_out")
    g = gi.combine_first(go)
    g[(gi.notna()) & (go.notna()) & (gi != go)] = np.nan
    return g

LEAK_PREFIX = ("min_bank", "q10_bank", "h_bank", "band0_bank", "band1_bank")
LAT_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=50,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, n_jobs=N_JOBS, deterministic=True, force_col_wise=True)

B = band_features(tx).reindex(F.index)
g = anchor(B)
XL = F.join(B)
lat_feat = [c for c in XL.columns if not c.startswith(LEAK_PREFIX)]
XLv = XL[lat_feat].astype(np.float32).values
m = g.notna().values
Xa, ta = XLv[m], g.values[m].astype(int)
oof_lat = np.zeros(m.sum()); pred_lat = np.zeros(len(XLv))
for sd in (0, 1, 2):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + sd).split(Xa, ta):
        mdl = lgb.train(dict(LAT_PARAMS, seed=sd), lgb.Dataset(Xa[a], ta[a]), 600)
        oof_lat[b] += mdl.predict(Xa[b]) / 3
        pred_lat += mdl.predict(XLv) / 15
lat_p = pd.Series(pred_lat, index=F.index); lat_p[m] = oof_lat
L = B.copy()
L["lat_p"] = lat_p; L["lat_anchor"] = g; L["lat_comb"] = g.fillna(lat_p)
print("якорей:", int(m.sum()), "| OOF AUC классификатора класса:", round(roc_auc_score(ta, oof_lat), 4))
yt = ys.reindex(F.index)
print("AUC эскалации по одному lat_p (train):", round(roc_auc_score(yt[yt.notna()], lat_p[yt.notna()]), 4),
      f"| {time.time() - T0:.0f}s")
""")

md("## 5. Модели: repeated stratified 5-fold × 3, одинаковые фолды для всех моделей")
code(r"""
X = F.join(L)
train_ids = S.signal_id[S.is_test == 0].values
test_ids = S.signal_id[S.is_test == 1].values
Y = S.set_index("signal_id").eskalatsiya.reindex(train_ids).values.astype(int)
A = X.reindex(train_ids).values.astype(np.float32)
T = X.reindex(test_ids).values.astype(np.float32)
FOLDS = [(r, f, a, b) for r in range(3)
         for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + r).split(A, Y))]

def fit_predict(kind, p, Xa, ya, Xs, seed):
    if kind == "lgb":
        mdl = lgb.train(dict(p, seed=seed), lgb.Dataset(Xa, ya), p["n_rounds"])
        return [mdl.predict(x) for x in Xs]
    if kind == "cat":
        q = {k: v for k, v in p.items() if k != "n_rounds"}
        mdl = CatBoostClassifier(iterations=p["n_rounds"], random_seed=seed, verbose=0, thread_count=N_JOBS, **q).fit(Xa, ya)
        return [mdl.predict_proba(x)[:, 1] for x in Xs]
    if kind == "xgb":
        q = {k: v for k, v in p.items() if k != "n_rounds"}
        mdl = xgb.train(dict(q, seed=seed, objective="binary:logistic", nthread=N_JOBS, tree_method="hist"),
                        xgb.DMatrix(Xa, ya), p["n_rounds"])
        return [mdl.predict(xgb.DMatrix(x)) for x in Xs]

def run(kind, p, seeds_per_fold=3):
    oofs, tests = [], []
    for r in range(3):
        oof = np.zeros(len(Y)); tp = np.zeros(len(T))
        for rr, f, a, b in FOLDS:
            if rr != r: continue
            for s in range(seeds_per_fold):
                pb, pt = fit_predict(kind, p, A[a], Y[a], [A[b], T], seed=1000 * r + 10 * f + s)
                oof[b] += pb / seeds_per_fold; tp += pt / (5 * seeds_per_fold)
        oofs.append(oof); tests.append(tp)
    aucs = [roc_auc_score(Y, o) for o in oofs]
    print(f"{kind}: {np.round(aucs, 4)} mean={np.mean(aucs):.4f}  ({time.time() - T0:.0f}s)")
    return np.array(oofs), np.mean(tests, 0)

PARAMS = {
    "lgb": dict(objective="binary", learning_rate=0.012, num_leaves=15, min_child_samples=500, feature_fraction=0.65,
                bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, min_gain_to_split=0.2, extra_trees=True,
                n_rounds=200, n_jobs=N_JOBS, verbose=-1, deterministic=True, force_col_wise=True),
    "cat": dict(n_rounds=400, learning_rate=0.02, depth=4, l2_leaf_reg=30, rsm=0.5,
                bootstrap_type="Bernoulli", subsample=0.7),
    "xgb": dict(n_rounds=250, eta=0.02, max_depth=4, min_child_weight=200, subsample=0.7,
                colsample_bytree=0.5, reg_lambda=10),
}
RES = {k: run(k, p) for k, p in PARAMS.items()}
""")

md("## 6. Бленд (среднее рангов) и сабмит")
code(r"""
blend_oof = [roc_auc_score(Y, sum(rankdata(RES[k][0][r]) for k in RES)) for r in range(3)]
print("blend OOF AUC:", np.round(blend_oof, 4), f"mean={np.mean(blend_oof):.4f}")

r = sum(rankdata(RES[k][1]) for k in RES)
r = (r - r.min()) / (r.max() - r.min())
sub = pd.DataFrame({"signal_id": test_ids, "ehtimollik": np.round(r, 6)})
test_order = pd.read_csv(DATA / "test_signals.csv").signal_id
sub = sub.set_index("signal_id").reindex(test_order).reset_index()
out = f"team_{TEAM_ID}.csv"
sub.to_csv(out, index=False)

chk = pd.read_csv(out)
assert list(chk.columns) == ["signal_id", "ehtimollik"]
assert len(chk) == len(test_order) == 6000 and chk.signal_id.is_unique and set(chk.signal_id) == set(test_order)
assert chk.ehtimollik.notna().all() and chk.ehtimollik.between(0, 1).all()
print("saved", out, chk.shape, f"total {time.time() - T0:.0f}s")
chk.head()
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "solution.ipynb")
print("written solution.ipynb with", len(cells), "cells")
