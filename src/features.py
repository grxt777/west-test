"""Признаки на уровне сигнала. Каждая группа — функция tx -> DataFrame(index=signal_id)."""
import numpy as np
import pandas as pd
from data import SEGS, TYPES, BURST_SEC, load_signals, load_tx, CACHE

DAY = 86400.0


def _flat(df, prefix=""):
    df.columns = [prefix + "_".join(str(c) for c in col) if isinstance(col, tuple) else prefix + str(col) for col in df.columns]
    return df


def g_profile(tx):
    """Параметры 'профиля клиента' по 8 сегментам: n, доля, mean, std, квантили; + относительно уровня клиента."""
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
    # уровень клиента: взвешенное среднее отклонений сегментов от их глобальных средних (по всем данным, без меток)
    gm = tx.groupby("seg").a.mean()
    dev = tx.a - tx.seg.map(gm)
    F["lvl_all"] = dev.groupby(tx.signal_id).mean()
    nonbank = tx.typ != 1
    F["lvl_nonbank"] = dev[nonbank].groupby(tx.signal_id[nonbank]).mean()
    F["lvl_card"] = dev[tx.typ == 0].groupby(tx.signal_id[tx.typ == 0]).mean()
    for ref in ["lvl_nonbank", "lvl_card"]:
        for sg_i, sg in enumerate(SEGS):
            F[f"rel_{ref}_{sg}"] = F[f"mean_{sg}"] - gm[sg_i] - F[ref]
    # парные разности средних сегментов
    base = ["karta_in", "karta_out", "bank_otkazmasi_in", "bank_otkazmasi_out", "naqd_in", "naqd_out"]
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            F[f"d_{a}__{b}"] = F[f"mean_{a}"] - F[f"mean_{b}"]
    F["bank_minus_card"] = (F.mean_bank_otkazmasi_in.fillna(F.mean_bank_otkazmasi_out) + F.mean_bank_otkazmasi_out.fillna(F.mean_bank_otkazmasi_in)) / 2 \
        - (F.mean_karta_in.fillna(F.mean_karta_out) + F.mean_karta_out.fillna(F.mean_karta_in)) / 2
    # отношение std сегментов к std карт
    for sg in SEGS[1:]:
        F[f"stdr_{sg}"] = F[f"std_{sg}"] / F["std_karta_in"]
    # внутриклиентская z-позиция: доля транзакций сегмента ниже уровня клиента
    F["a_mean"] = tx.groupby("signal_id").a.mean()
    F["a_std"] = tx.groupby("signal_id").a.std()
    return F


def g_time(tx, s):
    """Временная активность: окна до сигнала, всплеск, интервалы, час/день недели."""
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
    # наклон активности: регрессия log(1+cnt) по 30-дневным бинам
    b = np.clip((sec / (30 * DAY)).astype(int), 0, 5)
    c = pd.crosstab(tx.signal_id, b).reindex(F.index).fillna(0)
    lc = np.log1p(c.values); x = np.arange(6) - 2.5
    F["act_slope"] = (lc * x).sum(1) / (x ** 2).sum()
    F["first_day"] = tx.groupby("signal_id").sec.max() / DAY
    F["last_sec_nb"] = tx[tx.burst == 0].groupby("signal_id").sec.min()
    F["n_after_sig"] = (sec < 0).groupby(tx.signal_id).sum()
    # всплеск: состав и суммы
    bt = tx[tx.burst == 1]
    gb = bt.groupby("signal_id")
    F["burst_out"] = gb.out.mean(); F["burst_amean"] = gb.a.mean(); F["burst_astd"] = gb.a.std()
    for i, t in enumerate(TYPES):
        F[f"burst_frac_{t}"] = (bt.typ == i).groupby(bt.signal_id).mean()
    F["burst_span"] = gb.sec.max() - gb.sec.min()
    # интервалы (вне всплеска)
    nb = tx[tx.burst == 0]
    gap = nb.groupby("signal_id").sec.diff(-1).abs()  # отсортировано по времени: sec убывает
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
    """Денежные потоки: net-flow, баланс-прокси, pass-through (вход -> выход вскоре)."""
    F = pd.DataFrame(index=tx.signal_id.unique())
    v = np.exp(tx.a)  # индекс ~ лог-сумма: переходим к псевдо-суммам
    sgn = np.where(tx.out == 1, -1.0, 1.0)
    F["net_idx"] = (tx.a * sgn).groupby(tx.signal_id).sum()
    F["in_vol"] = (v * (1 - tx.out)).groupby(tx.signal_id).sum()
    F["out_vol"] = (v * tx.out).groupby(tx.signal_id).sum()
    F["out_in_ratio"] = F.out_vol / F.in_vol
    bal = (v * sgn).groupby(tx.signal_id).cumsum()
    gb = bal.groupby(tx.signal_id)
    F["bal_min"] = gb.min() / F.in_vol; F["bal_max"] = gb.max() / F.in_vol
    # pass-through: для каждой исходящей — время с последней входящей
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
    # наличные снятия после входящего банковского перевода
    t["t_bin"] = np.where((t.out == 0) & (t.typ == 1), -t.sec, np.nan); t["t_bin"] = t.groupby("signal_id").t_bin.ffill()
    co = t[(t.out == 1) & (t.typ == 2)]
    F["cash_after_bank_1d"] = ((-co.sec - co.t_bin) < DAY).groupby(co.signal_id).mean()
    return F


def g_hist(tx, nb=12):
    """Гистограмма сумм внутри сегмента по глобальным квантилям сегмента (доли)."""
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


def build_all(force=False):
    f = CACHE / "feat_v2.parquet"
    if f.exists() and not force:
        return pd.read_parquet(f)
    s = load_signals(); tx = load_tx()
    F = pd.concat([g_profile(tx), g_time(tx, s), g_flow(tx), g_hist(tx)], axis=1).reindex(s.signal_id)
    F = F.replace([np.inf, -np.inf], np.nan)
    F.to_parquet(f)
    return F


if __name__ == "__main__":
    import time; t0 = time.time()
    F = build_all(force=True); print(F.shape, round(time.time() - t0), "s")
