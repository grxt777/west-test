"""Скрытый класс клиента (A/B) — главный механизм генератора данных.

Найдено в EDA: у каждого класса в каждом сегменте (тип x направление) свой жёсткий минимум суммы
(смещение от глобального минимума сегмента). Значения ниже минимума класса 'прижаты' в узкую полосу
над ним. Поэтому транзакция в зоне [floor, минимум другого класса) однозначно определяет класс.
Проверка: 6045 клиентов определяются правилом, 0 противоречий. Эскалация: A ~10%, B ~30%.

Строим (таргет НЕ используется, train+test вместе):
  ev_*     — число транзакций в 'запретных' зонах (детерминированные улики);
  det      — класс по правилу (0=A, 1=B, NaN — не определён);
  lam_*    — ожидаемое число транзакций в запретной зоне при 'другом' классе (сила улики отсутствия);
  llr_floor— суммарный лог-отношение правдоподобий B:A по отсутствию/наличию улик;
  lat_p    — вероятность класса B от классификатора 'формы' профиля, обученного на det-якорях
             (OOF для якорей, без признаков, прямо раскрывающих правило);
  lat_post — итоговая апостериорная вероятность класса B (det, иначе lat_p скорректированная llr_floor).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from data import SEGS, CACHE, load_tx

# минимум класса (смещение от глобального минимума сегмента): seg -> (A_min, B_min)
CLASS_FLOOR = {0: (0.0357, 0.0), 1: (0.1431, 0.0), 2: (0.0, 0.0178), 3: (0.0, 0.0318),
               4: (0.0, 0.0407), 5: (0.0, 0.1276), 6: (0.0, 0.0995), 7: (0.4545, 0.0)}


def evidence(tx):
    fl = tx.groupby("seg").a.min()
    off = tx.a - tx.seg.map(fl)
    sid = tx.signal_id
    out = {}
    for sg, (fa, fb) in CLASS_FLOOR.items():
        m_sg = tx.seg == sg
        if fa > fb:   # зона [0, A_min) доступна только B
            out[f"evB_{SEGS[sg]}"] = (m_sg & (off < fa)).groupby(sid).sum()
        else:         # зона [0, B_min) доступна только A
            out[f"evA_{SEGS[sg]}"] = (m_sg & (off < fb)).groupby(sid).sum()
        # плотность у минимума каждого класса (зона 'прижатия')
        out[f"pileA_{SEGS[sg]}"] = (m_sg & (off >= fa) & (off < fa + 0.03)).groupby(sid).sum()
        out[f"pileB_{SEGS[sg]}"] = (m_sg & (off >= fb) & (off < fb + 0.03)).groupby(sid).sum()
        # робастные параметры клиента в сегменте (по верхней части, не задетой порогом)
        g = (tx.a[m_sg]).groupby(sid[m_sg])
        q = g.quantile([.5, .75, .9]).unstack()
        sig = ((q[.9] - q[.5]) / 1.2816 + (q[.75] - q[.5]) / 0.6745) / 2
        n = g.size()
        thr = fl[sg] + max(fa, fb)
        lam = n * norm.cdf((thr - q[.5]) / sig.clip(lower=0.05))
        out[f"lam_{SEGS[sg]}"] = lam
    E = pd.DataFrame(out)
    return E


def det_class(E):
    eB = E.filter(like="evB_").fillna(0).sum(1)
    eA = E.filter(like="evA_").fillna(0).sum(1)
    d = pd.Series(np.nan, index=E.index)
    d[(eB > 0) & (eA == 0)] = 1
    d[(eA > 0) & (eB == 0)] = 0
    return d


def floor_llr(E):
    """log P(наблюдения по зонам | B) - log P(... | A), с насыщением при наличии улики."""
    llr = pd.Series(0.0, index=E.index)
    for sg, (fa, fb) in CLASS_FLOOR.items():
        s = SEGS[sg]
        lam = E[f"lam_{s}"].fillna(0)
        if fa > fb:  # отсутствие B-улики -> в пользу A
            ev = E[f"evB_{s}"].fillna(0)
            llr += np.where(ev > 0, 20.0, -lam)
        else:
            ev = E[f"evA_{s}"].fillna(0)
            llr += np.where(ev > 0, -20.0, lam)
    return llr.clip(-20, 20)


# признаки, раскрывающие правило (минимумы, нижние квантили, гистограммы, полосы, улики)
LEAK_PREFIX = ("min_", "q10_", "h_", "band", "ev", "pile", "lam_", "llr", "low", "clip")

LAT_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=50,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, n_jobs=8)


def shape_classifier(F, d, n_rounds=700, seeds=(0, 1, 2)):
    feat = [c for c in F.columns if not c.startswith(LEAK_PREFIX)]
    X = F[feat].astype(np.float32).values
    m = d.notna().values
    Xa, ta = X[m], d.values[m].astype(int)
    oof = np.zeros(m.sum()); pred = np.zeros(len(X))
    for sd in seeds:
        for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + sd).split(Xa, ta):
            mdl = lgb.train(dict(LAT_PARAMS, seed=sd), lgb.Dataset(Xa[a], ta[a]), n_rounds)
            oof[b] += mdl.predict(Xa[b]) / len(seeds)
            pred += mdl.predict(X) / (5 * len(seeds))
    p = pd.Series(pred, index=F.index)
    p[m] = oof
    return p, (ta, oof)


def build_latent(F, force=False, verbose=True):
    f = CACHE / "latent_v2.parquet"
    if f.exists() and not force:
        return pd.read_parquet(f)
    E = evidence(load_tx()).reindex(F.index)
    d = det_class(E)
    p, (ta, oof) = shape_classifier(F, d)
    llr = floor_llr(E)
    lp = np.log(p.clip(1e-4, 1 - 1e-4) / (1 - p.clip(1e-4, 1 - 1e-4)))
    post = 1 / (1 + np.exp(-(lp + llr)))
    post[d.notna()] = d[d.notna()]
    out = E.copy()
    out["det"] = d; out["llr_floor"] = llr; out["lat_p"] = p; out["lat_post"] = post
    out["lat_comb"] = d.fillna(p)
    if verbose:
        from sklearn.metrics import roc_auc_score
        print("det coverage:", int(d.notna().sum()), "| shape classifier OOF AUC on det anchors:",
              round(roc_auc_score(ta, oof), 4))
    out.to_parquet(f)
    return out
