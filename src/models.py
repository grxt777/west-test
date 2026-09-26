"""Обучение моделей на фиксированных repeated-фолдах: OOF по train + усреднённый прогноз на test."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from data import folds, CACHE

IDS, Y, FOLDS = folds(n_rep=3, k=5)


def _fit_predict(kind, params, Xa, ya, Xb_list, seed):
    if kind == "lgb":
        m = lgb.train(dict(params, seed=seed, verbose=-1), lgb.Dataset(Xa, ya), params["n_rounds"])
        return [m.predict(X) for X in Xb_list]
    if kind == "cat":
        from catboost import CatBoostClassifier
        p = {k: v for k, v in params.items() if k != "n_rounds"}
        m = CatBoostClassifier(iterations=params["n_rounds"], random_seed=seed, verbose=0, thread_count=8, **p)
        m.fit(Xa, ya)
        return [m.predict_proba(X)[:, 1] for X in Xb_list]
    if kind == "xgb":
        import xgboost as xgb
        p = {k: v for k, v in params.items() if k != "n_rounds"}
        m = xgb.train(dict(p, seed=seed, objective="binary:logistic", eval_metric="auc", nthread=8, tree_method="hist"),
                      xgb.DMatrix(Xa, ya), params["n_rounds"])
        return [m.predict(xgb.DMatrix(X)) for X in Xb_list]
    raise ValueError(kind)


def run(kind, params, Xtr, Xte, reps=(0, 1, 2), seeds_per_fold=1, name=None, verbose=True):
    """Xtr — DataFrame в порядке IDS, Xte — DataFrame test. Возвращает dict(oof, test, aucs)."""
    A = Xtr.values.astype(np.float32); T = Xte.values.astype(np.float32)
    oofs, tests, aucs = [], [], []
    for r in reps:
        oof = np.zeros(len(Y)); te = np.zeros(len(T))
        for rr, f, a, b in FOLDS:
            if rr != r:
                continue
            for s in range(seeds_per_fold):
                pb, pt = _fit_predict(kind, params, A[a], Y[a], [A[b], T], seed=1000 * r + 10 * f + s)
                oof[b] += pb / seeds_per_fold; te += pt / (5 * seeds_per_fold)
        oofs.append(oof); tests.append(te); aucs.append(roc_auc_score(Y, oof))
    res = dict(oof=np.mean(oofs, 0), test=np.mean(tests, 0), aucs=np.array(aucs),
               mean=float(np.mean(aucs)), oof_reps=oofs)
    if verbose:
        print(f"{name or kind}: {np.round(aucs, 4)} mean={res['mean']:.4f}")
    if name:
        pd.DataFrame({"signal_id": IDS, "oof": res["oof"]}).to_parquet(CACHE / f"oof_{name}.parquet")
        pd.DataFrame({"signal_id": Xte.index, "pred": res["test"]}).to_parquet(CACHE / f"test_{name}.parquet")
        np.save(CACHE / f"oofreps_{name}.npy", np.array(oofs))
    return res
