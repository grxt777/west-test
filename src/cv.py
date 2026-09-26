"""Repeated stratified CV для LightGBM/CatBoost/XGB на одних и тех же фолдах."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from data import folds

IDS, Y, FOLDS = folds(n_rep=3, k=5)

LGB_BASE = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=100,
                feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                verbose=-1, n_jobs=8, seed=0)


def run_lgb(X, params=None, rounds=3000, es=200, reps=(0, 1, 2), ret_oof=False):
    """X: DataFrame по train-сигналам в порядке IDS. Возвращает AUC по повторам (+ oof)."""
    p = dict(LGB_BASE, **(params or {}))
    X = X.values.astype(np.float32) if hasattr(X, "values") else X
    oofs, aucs, iters = [], [], []
    for r in reps:
        oof = np.zeros(len(Y))
        for rr, f, a, b in FOLDS:
            if rr != r:
                continue
            dtr = lgb.Dataset(X[a], Y[a]); dva = lgb.Dataset(X[b], Y[b])
            m = lgb.train(dict(p, seed=p["seed"] + f), dtr, rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(es, verbose=False)])
            oof[b] = m.predict(X[b], num_iteration=m.best_iteration); iters.append(m.best_iteration)
        oofs.append(oof); aucs.append(roc_auc_score(Y, oof))
    res = dict(mean=float(np.mean(aucs)), std=float(np.std(aucs)), aucs=np.round(aucs, 4), iters=int(np.mean(iters)))
    if ret_oof:
        res["oof"] = np.mean(oofs, 0)
    return res
