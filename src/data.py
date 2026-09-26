"""Загрузка данных и общие константы."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fintech_data"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
SEGS = [f"{t}_{d}" for t in TYPES for d in ("in", "out")]  # seg = type*2 + out
BURST_SEC = 600  # транзакции за последние 10 минут до signal_sanasi = "всплеск"


def load_signals():
    tr = pd.read_csv(DATA / "train_signals.csv", parse_dates=["signal_sanasi"])
    te = pd.read_csv(DATA / "test_signals.csv", parse_dates=["signal_sanasi"])
    tr["is_test"] = 0
    te["is_test"] = 1
    te["eskalatsiya"] = np.nan
    s = pd.concat([tr, te], ignore_index=True)
    return s.sort_values("signal_id").reset_index(drop=True)


def load_tx():
    """Все транзакции train+test с предрассчитанными полями (кэш)."""
    f = CACHE / "tx_all.parquet"
    if f.exists():
        return pd.read_parquet(f)
    s = load_signals()[["signal_id", "signal_sanasi"]]
    tx = pd.concat([pd.read_parquet(DATA / "train_transactions.parquet"),
                    pd.read_parquet(DATA / "test_transactions.parquet")], ignore_index=True)
    tx = tx.merge(s, on="signal_id", how="left")
    tx["sec"] = (tx.signal_sanasi - tx.tranzaksiya_vaqti).dt.total_seconds().astype("float64")
    tx["a"] = tx.miqdor_indeksi.astype("float64")
    tx["out"] = (tx.kirim_chiqim == "chiqim").astype("int8")
    tx["typ"] = tx.tranzaksiya_turi.map({t: i for i, t in enumerate(TYPES)}).astype("int8")
    tx["seg"] = (tx.typ * 2 + tx.out).astype("int8")
    tx["burst"] = (tx.sec <= BURST_SEC).astype("int8")
    tx["hour"] = tx.tranzaksiya_vaqti.dt.hour.astype("int8")
    tx["dow"] = tx.tranzaksiya_vaqti.dt.dayofweek.astype("int8")
    tx = tx.sort_values(["signal_id", "tranzaksiya_vaqti", "a"], kind="mergesort").reset_index(drop=True)
    tx = tx[["signal_id", "tranzaksiya_vaqti", "sec", "a", "out", "typ", "seg", "burst", "hour", "dow"]]
    tx.to_parquet(f)
    return tx


def folds(n_rep=3, k=5):
    """Фиксированные repeated stratified фолды по train-сигналам: список (rep, fold, tr_idx, va_idx)."""
    from sklearn.model_selection import StratifiedKFold
    s = load_signals()
    tr = s[s.is_test == 0].reset_index(drop=True)
    y = tr.eskalatsiya.values.astype(int)
    out = []
    for r in range(n_rep):
        for f, (a, b) in enumerate(StratifiedKFold(k, shuffle=True, random_state=42 + r).split(tr, y)):
            out.append((r, f, a, b))
    return tr.signal_id.values, y, out
