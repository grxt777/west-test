"""Готовит site/data.json — данные для графиков EDA-сайта."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from data import ROOT, SEGS, CACHE, load_signals, load_tx
from features import build_all, DAY

s = load_signals(); tx = load_tx(); F = build_all()
L1 = pd.read_parquet(CACHE / "latent.parquet"); E = pd.read_parquet(CACHE / "evidence.parquet")
ys = s.set_index("signal_id").eskalatsiya; trm = s.is_test == 0; tr = s[trm]
D = {}
D["kpi"] = dict(n_train=int(trm.sum()), n_test=int((~trm).sum()), n_tx=int(len(tx)), pos=int(tr.eskalatsiya.sum()),
                rate=round(float(tr.eskalatsiya.mean()), 4), tx_med=int(tx.groupby("signal_id").size().median()))
m = tr.groupby(tr.signal_sanasi.dt.to_period("M")).eskalatsiya.agg(["mean", "size"])
D["monthly"] = [dict(m=str(k), rate=round(float(v["mean"]), 4), n=int(v["size"])) for k, v in m.iterrows()]
d = np.floor(tx.sec / DAY).clip(-1, 179).astype(int); c = d.value_counts().sort_index()
D["daily"] = [dict(d=int(k), n=int(v)) for k, v in c.items() if k >= 0]
b = tx.sec[(tx.sec > 0) & (tx.sec <= 600)]; h, e = np.histogram(b, bins=np.arange(0, 601, 10))
D["burst"] = [dict(s=int(e[i]), n=int(h[i])) for i in range(len(h))]
D["burst_share"] = round(float((tx.sec <= 600).mean()), 4)
hh = tx.hour.value_counts(normalize=True).sort_index(); D["hour"] = [dict(h=int(k), p=round(float(v), 4)) for k, v in hh.items()]
g = tx.groupby("seg").a.agg(["size", "mean", "std"])
D["segs"] = [dict(seg=SEGS[k], n=int(v["size"]), share=round(float(v["size"] / len(tx)), 4), mean=round(float(v["mean"]), 3),
                  std=round(float(v["std"]), 3)) for k, v in g.iterrows()]
t = tx[tx.signal_id.isin(tr.signal_id)]; yy = t.signal_id.map(ys).values
bins = np.arange(-3, 5.01, 0.2); h0 = np.histogram(t.a[yy == 0], bins)[0]; h1 = np.histogram(t.a[yy == 1], bins)[0]
D["amt_hist"] = [dict(x=round(float(bins[i] + 0.1), 2), p0=round(float(h0[i] / h0.sum()), 5), p1=round(float(h1[i] / h1.sum()), 5)) for i in range(len(h0))]
seg_y = []
for sg in range(6):
    tt = t[t.seg == sg]; y2 = tt.signal_id.map(ys)
    seg_y.append(dict(seg=SEGS[sg], m0=round(float(tt.a[y2 == 0].median()), 3), m1=round(float(tt.a[y2 == 1].median()), 3)))
D["seg_med_by_y"] = seg_y
C = F[[f"mean_{x}" for x in SEGS[:6]]].corr().round(2); D["corr"] = dict(labels=SEGS[:6], m=C.values.tolist())
fl = tx.groupby("seg").a.min(); tx["off"] = tx.a - tx.seg.map(fl)
eB = E.filter(like="ev1_").sum(1); eA = E.filter(like="ev0_").sum(1)
det = pd.Series(np.nan, index=E.index); det[(eB > 0) & (eA == 0)] = 1; det[(eA > 0) & (eB == 0)] = 0
tx["cls"] = tx.signal_id.map(det); tx["y"] = tx.signal_id.map(ys)


def floorzoom(sg, w, st):
    dd = tx[tx.seg == sg]; e = np.arange(0, w + 1e-9, st)
    all_ = np.histogram(dd.off, e)[0]
    b = np.searchsorted(e, dd.off.values, side="right") - 1; ok = (b >= 0) & (b < len(e) - 1) & dd.y.notna().values
    r = pd.Series(dd.y.values[ok]).groupby(b[ok]).mean()
    hA = np.histogram(dd.off[dd.cls == 0], e)[0]; hB = np.histogram(dd.off[dd.cls == 1], e)[0]
    return [dict(x=round(float(e[i]), 3), n=int(all_[i]), A=int(hA[i]), B=int(hB[i]),
                 rate=(round(float(r[i]), 3) if i in r.index else None)) for i in range(len(all_))]


D["floor_bank_in"] = floorzoom(2, 0.1, 0.005); D["floor_karta_out"] = floorzoom(1, 0.3, 0.01)
D["class_floor"] = [dict(seg=SEGS[k], A=v[0], B=v[1]) for k, v in {0: (0.0357, 0.0), 1: (0.1431, 0.0), 2: (0.0, 0.0178), 3: (0.0, 0.0318),
                    4: (0.0, 0.0407), 5: (0.0, 0.1276), 6: (0.0, 0.0995), 7: (0.4545, 0.0)}.items()]
yt = ys.reindex(det.index)
D["det"] = dict(nA=int((det == 0).sum()), nB=int((det == 1).sum()), conf=int(((eA > 0) & (eB > 0)).sum()),
                rateA=round(float(yt[det == 0].mean()), 3), rateB=round(float(yt[det == 1].mean()), 3),
                rateU=round(float(yt[det.isna()].mean()), 3), nU=int(det.isna().sum()))
lp = L1.lat_p.reindex(tr.signal_id).values; yv = tr.eskalatsiya.values; q = pd.qcut(lp, 20, labels=False)
D["lat_curve"] = [dict(p=round(float(lp[q == i].mean()), 3), rate=round(float(yv[q == i].mean()), 3), n=int((q == i).sum())) for i in range(20)]
D["progress"] = [dict(k="Бейзлайн: 140 агрегатов", v=0.6370), dict(k="+ профиль по 8 сегментам", v=0.6447),
                 dict(k="+ минимумы сегментов", v=0.6555), dict(k="+ скрытый класс", v=0.6648),
                 dict(k="Честно: фикс. итерации + тюнинг", v=0.6641), dict(k="Бленд LGB + Cat + XGB", v=0.6643)]
D["models"] = [dict(k="Бленд (ранги)", v=0.6643), dict(k="CatBoost", v=0.6642), dict(k="LightGBM", v=0.6641),
               dict(k="XGBoost", v=0.6631), dict(k="LightGBM DART", v=0.6615), dict(k="Логистическая регрессия", v=0.6461)]
D["ablation"] = [dict(k="все признаки", v=0.6555), dict(k="без потоков", v=0.6551), dict(k="без гистограмм", v=0.6549),
                 dict(k="без времени", v=0.6547), dict(k="без минимумов", v=0.6447), dict(k="без профиля", v=0.6225)]
X = F.join(L1); ids = tr.signal_id.values
mdl = lgb.train(dict(objective="binary", learning_rate=0.012, num_leaves=15, min_child_samples=500, feature_fraction=0.65,
                     bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, min_gain_to_split=0.2, extra_trees=True,
                     verbose=-1, seed=0, deterministic=True),
                lgb.Dataset(X.reindex(ids).values.astype(np.float32), tr.eskalatsiya.values), 200)
imp = pd.Series(mdl.feature_importance("gain"), index=X.columns); imp = imp / imp.sum()
D["imp"] = [dict(k=k, v=round(float(v), 4)) for k, v in imp.sort_values(ascending=False).head(15).items()]
json.dump(D, open(ROOT / "site" / "data.json", "w"), ensure_ascii=False)
print({k: (len(v) if isinstance(v, list) else v) for k, v in D.items() if k != "corr"})
