#!/usr/bin/env python3
"""Ridge logistic regression baseline, F1/PR-AUC for the baselines,
and side-by-side XGBoost-gain / ridge-coefficient variable importance.
Outputs:
    results/holistic_eval.json
    results/var_importance_T{1,2,3,4}.csv
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, brier_score_loss,
)

warnings.filterwarnings("ignore")

from config import (
    PROCESSED_DIR, RESULTS_DIR, TASKS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight,
    get_feature_columns,
)

EPS = 1e-12
COST_MISS = 25.0
COST_FA = 1.0
TASK_IDS = ["T1", "T2", "T3", "T4"]

def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "aerocalib_hourly.parquet")
    df["year"] = df["datetime"].dt.year
    return df

def split_xy(df, feature_cols, label_col, year_lo, year_hi):
    sub = df[(df.year >= year_lo) & (df.year <= year_hi)].dropna(subset=[label_col])
    X = sub[feature_cols].values.astype(np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    y = sub[label_col].values.astype(np.int8)
    return X, y

def impute_median(X_train, *Xs):
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def fill(X):
        Xi = X.copy()
        for j in range(X.shape[1]):
            mask = ~np.isfinite(Xi[:, j])
            if mask.any():
                Xi[mask, j] = med[j]
        return Xi
    return (fill(X_train),) + tuple(fill(X) for X in Xs)

def f1_threshold_on_val(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / np.maximum(prec + rec, EPS)
    idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    t = float(thr[idx]) if idx < len(thr) else 0.5
    return t

def evaluate(name, y_test, p_test, y_val, p_val):
    thr = f1_threshold_on_val(y_val, p_val)
    return {
        "name": name,
        "n_test": int(len(y_test)),
        "n_pos_test": int(y_test.sum()),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "brier": float(brier_score_loss(y_test, p_test)),
        "thr_best_f1_val": thr,
        "f1_at_best_thr": float(f1_score(y_test, (p_test >= thr).astype(int),
                                          zero_division=0)),
        "f1_at_cost_thr": float(f1_score(
            y_test,
            (p_test >= COST_FA / (COST_MISS + COST_FA)).astype(int),
            zero_division=0,
        )),
    }

def run_task(df, task_id, feature_cols):
    label_col = f"{task_id}_h{TASKS[task_id].horizon}"
    print(f"\nTask {task_id}  ({label_col})")

    X_tr, y_tr = split_xy(df, feature_cols, label_col, TRAIN_YEARS[0], TRAIN_YEARS[1])
    X_va, y_va = split_xy(df, feature_cols, label_col, VAL_YEARS[0], VAL_YEARS[1])
    X_te, y_te = split_xy(df, feature_cols, label_col, TEST_YEARS[0], TEST_YEARS[1])
    print(f"  train={len(y_tr):,} val={len(y_va):,} test={len(y_te):,}")
    print(f"  positives train={int(y_tr.sum())} val={int(y_va.sum())} test={int(y_te.sum())}")

    X_tr_i, X_va_i, X_te_i = impute_median(X_tr, X_va, X_te)
    scaler = StandardScaler().fit(X_tr_i)
    X_tr_s = scaler.transform(X_tr_i)
    X_va_s = scaler.transform(X_va_i)
    X_te_s = scaler.transform(X_te_i)

    results = {}

    print("  LR (no penalty)")
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500,
                            class_weight="balanced", n_jobs=-1)
    lr.fit(X_tr_s, y_tr)
    results["LR_plain"] = evaluate("LR (no penalty)",
                                    y_te, lr.predict_proba(X_te_s)[:, 1],
                                    y_va, lr.predict_proba(X_va_s)[:, 1])

    print("  LR (ridge, C=1)")
    ridge = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500,
                               class_weight="balanced", n_jobs=-1)
    ridge.fit(X_tr_s, y_tr)
    results["LR_ridge"] = evaluate("LR (ridge)",
                                    y_te, ridge.predict_proba(X_te_s)[:, 1],
                                    y_va, ridge.predict_proba(X_va_s)[:, 1])

    ridge_coefs = pd.DataFrame({
        "feature": feature_cols,
        "ridge_coef_std": ridge.coef_[0],
        "abs_coef": np.abs(ridge.coef_[0]),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    print("  Random Forest")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20,
                                class_weight="balanced", n_jobs=-1,
                                random_state=RANDOM_SEED)
    rf.fit(X_tr_i, y_tr)
    results["RF"] = evaluate("Random Forest",
                              y_te, rf.predict_proba(X_te_i)[:, 1],
                              y_va, rf.predict_proba(X_va_i)[:, 1])

    print("  XGBoost")
    params = {k: v for k, v in XGBOOST_PARAMS.items()
              if k not in ("early_stopping_rounds", "random_state")}
    params["scale_pos_weight"] = compute_scale_pos_weight(float(y_tr.mean()))
    xgbm = xgb.XGBClassifier(**params, random_state=RANDOM_SEED,
                              use_label_encoder=False)
    xgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    results["XGBoost"] = evaluate("XGBoost",
                                   y_te, xgbm.predict_proba(X_te)[:, 1],
                                   y_va, xgbm.predict_proba(X_va)[:, 1])

    booster = xgbm.get_booster()
    gain = booster.get_score(importance_type="gain")
    fnames = {f"f{i}": feature_cols[i] for i in range(len(feature_cols))}
    xg_imp = pd.DataFrame(
        [{"feature": fnames.get(k, k), "xgb_gain": v} for k, v in gain.items()]
    ).sort_values("xgb_gain", ascending=False).reset_index(drop=True)

    var_imp = ridge_coefs.merge(xg_imp, on="feature", how="outer").fillna(0.0)
    var_imp["xgb_gain_norm"] = var_imp["xgb_gain"] / max(var_imp["xgb_gain"].sum(), EPS)
    var_imp["ridge_abs_norm"] = var_imp["abs_coef"] / max(var_imp["abs_coef"].sum(), EPS)
    var_imp = var_imp.sort_values("xgb_gain", ascending=False).reset_index(drop=True)

    out_csv = RESULTS_DIR / f"var_importance_{task_id}.csv"
    var_imp.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")

    return results, var_imp

def print_latex_baseline_block(all_results):
    print("\nBaseline LaTeX rows")
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1_at_best_thr", "F1")]
    keys = [("LR_plain", "Logistic Regression"),
            ("LR_ridge", "Ridge Logistic Regression"),
            ("RF", "Random Forest"),
            ("XGBoost", "XGBoost")]
    for metric_key, metric_lbl in metrics:
        print(f"% {metric_lbl}")
        for key, name in keys:
            row = [all_results["tasks"][t][key][metric_key] for t in TASK_IDS]
            bold = name == "XGBoost"
            cells = " & ".join(
                (f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}") for v in row
            )
            print(f"{name} & {cells} \\\\")

def print_latex_varimp_block(var_imp_all, n=10):
    print("\nVariable importance LaTeX rows")
    for task_id in TASK_IDS:
        print(f"% Task {task_id}")
        for _, r in var_imp_all[task_id].head(n).iterrows():
            feat = str(r["feature"]).replace("_", r"\_")
            print(f"{feat} & {r['xgb_gain_norm']:.3f} & {r['ridge_coef_std']:+.3f} \\\\")

def main():
    df = load_data()
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    print(f"{len(df):,} rows, {len(feature_cols)} features")

    all_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "22_holistic_eval.py",
            "cost_ratio_miss_fa": [COST_MISS, COST_FA],
            "cost_optimal_threshold": COST_FA / (COST_MISS + COST_FA),
            "n_features": len(feature_cols),
        },
        "tasks": {},
    }
    var_imp_all = {}
    for task_id in TASK_IDS:
        results, var_imp = run_task(df, task_id, feature_cols)
        all_results["tasks"][task_id] = results
        var_imp_all[task_id] = var_imp

    out_json = RESULTS_DIR / "holistic_eval.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n-> {out_json}")

    print_latex_baseline_block(all_results)
    print_latex_varimp_block(var_imp_all)

    for task_id in TASK_IDS:
        r = all_results["tasks"][task_id]
        d_auc = r["XGBoost"]["roc_auc"] - r["LR_ridge"]["roc_auc"]
        d_pr = r["XGBoost"]["pr_auc"] - r["LR_ridge"]["pr_auc"]
        print(f"\n{task_id}  XGB-Ridge: dAUC={d_auc:+.3f}  dPR-AUC={d_pr:+.3f}")

if __name__ == "__main__":
    main()
