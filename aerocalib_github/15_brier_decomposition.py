#!/usr/bin/env python3
"""Computes Murphy (1973) Brier score decomposition:."""

import json
import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    TASKS, VAL_YEARS, TEST_YEARS,
    ECE_N_BINS, RANDOM_SEED, get_feature_columns,
    print_memory
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
# Calibration: Platt scaling

def platt_scaling(y_val, scores_val, scores_test, clip_eps=1e-10):
    """
    Fit Platt scaling on validation labels and raw probs, apply to test probs.

    Returns:
      calibrated_test, A, B, status_str
    """
    y_val = np.asarray(y_val, dtype=float)
    scores_val = np.asarray(scores_val, dtype=float)
    scores_test = np.asarray(scores_test, dtype=float)

    # If no class variation -> can't fit logistic regression meaningfully
    uniq = np.unique(y_val)
    if uniq.size < 2:
        return scores_test.astype(np.float32), 1.0, 0.0, "fallback_identity_no_class_variation"

    # Clip for numerical stability
    scores_val = np.clip(scores_val, clip_eps, 1.0 - clip_eps)
    logits_val = logit(scores_val)

    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
    try:
        lr.fit(logits_val.reshape(-1, 1), y_val)
    except Exception:
        del lr
        return scores_test.astype(np.float32), 1.0, 0.0, "fallback_identity_fit_failed"

    A = float(lr.coef_[0][0])
    B = float(lr.intercept_[0])
    del lr

    scores_test = np.clip(scores_test, clip_eps, 1.0 - clip_eps)
    logits_test = logit(scores_test)
    calibrated_test = expit(A * logits_test + B).astype(np.float32)

    return calibrated_test, A, B, "ok"
# Murphy (1973) Brier score decomposition

def brier_decomposition(y_true, y_prob, n_bins=10, strategy="quantile"):
    """
    Murphy (1973) decomposition:
      Brier = Reliability - Resolution + Uncertainty

    Returns dict:
      brier, reliability, resolution, uncertainty, n_bins_used
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    n = int(len(y_true))
    if n == 0:
        return {
            "brier": float("nan"),
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
            "n_bins_used": 0,
        }

    bar_o = float(y_true.mean())  # overall event frequency
    uncertainty = bar_o * (1.0 - bar_o)

    # Bin edges
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, qs)
        bin_edges = np.unique(bin_edges)
        # If edges collapse too much, fallback to uniform bins
        if bin_edges.size < 3:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Digitize: bins are 0..K-2
    bin_idx = np.digitize(y_prob, bin_edges[1:-1], right=False)

    reliability = 0.0
    resolution = 0.0
    n_bins_used = 0

    for i in range(bin_edges.size - 1):
        mask = (bin_idx == i)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        n_bins_used += 1

        bar_o_k = float(y_true[mask].mean())
        bar_f_k = float(y_prob[mask].mean())

        reliability += n_k * (bar_f_k - bar_o_k) ** 2
        resolution += n_k * (bar_o_k - bar_o) ** 2

    reliability /= n
    resolution /= n
    brier = reliability - resolution + uncertainty

    return {
        "brier": float(brier),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "n_bins_used": int(n_bins_used),
    }
# Data loading helpers

def load_data_for_task(task_id):
    parquet_path = PROCESSED_DIR / "aerocalib_hourly.parquet"
    feature_cols = get_feature_columns()

    task_config = TASKS[task_id]
    label_col = f"{task_id}_h{task_config.horizon}"

    needed_cols = feature_cols + [label_col, "datetime"]

    try:
        df = pd.read_parquet(parquet_path, columns=needed_cols + ["year"])
    except Exception:
        df = pd.read_parquet(parquet_path, columns=needed_cols)
        # ensure datetime dtype
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["year"] = df["datetime"].dt.year

    val_mask = (df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1])
    test_mask = (df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1])

    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    del df
    gc.collect()
    return val_df, test_df

def decompose_task(task_id):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f"{task_id}_h{horizon}"
    feature_cols = get_feature_columns()

    print(f"\n{'='*60}")
    print(f"Brier Decomposition: {task_id} ({task_config.name})")
    print(f"{'='*60}")

    # Load model
    model_path = MODELS_DIR / f"xgb_{task_id}_h{horizon}.json"
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    # Load data
    val_df, test_df = load_data_for_task(task_id)
    available_features = [f for f in feature_cols if f in val_df.columns]

    val_data = val_df.dropna(subset=[label_col])
    test_data = test_df.dropna(subset=[label_col])

    X_val = val_data[available_features].values.astype(np.float32)
    y_val = val_data[label_col].values.astype(np.float32)
    X_test = test_data[available_features].values.astype(np.float32)
    y_test = test_data[label_col].values.astype(np.float32)

    del val_df, test_df, val_data, test_data
    gc.collect()

    # Raw predictions
    y_prob_val_raw = model.predict_proba(X_val)[:, 1].astype(np.float32)
    y_prob_test_raw = model.predict_proba(X_test)[:, 1].astype(np.float32)

    del model, X_val, X_test
    gc.collect()

    # Platt scaling (fit on val, apply to test)
    y_prob_test_platt, A, B, status = platt_scaling(
        y_val=y_val,
        scores_val=y_prob_val_raw,
        scores_test=y_prob_test_raw,
        clip_eps=1e-10
    )

    # Decomposition (same binning as ECE)
    pre_decomp = brier_decomposition(y_test, y_prob_test_raw, n_bins=ECE_N_BINS, strategy="quantile")
    post_decomp = brier_decomposition(y_test, y_prob_test_platt, n_bins=ECE_N_BINS, strategy="quantile")
    nll_post = log_loss(y_test, np.clip(y_prob_test_platt, 1e-10, 1 - 1e-10))

    print(f"  Prevalence: {y_test.mean()*100:.3f}%")
    print(f"  Platt status: {status} (A={A:.6f}, B={B:.6f})")
    print("  Post-calibration Brier decomposition (Murphy 1973):")
    print(f"    Brier       = {post_decomp['brier']:.6f}")
    print(f"    Reliability = {post_decomp['reliability']:.6f}")
    print(f"    Resolution  = {post_decomp['resolution']:.6f}")
    print(f"    Uncertainty = {post_decomp['uncertainty']:.6f}")
    check_val = post_decomp["reliability"] - post_decomp["resolution"] + post_decomp["uncertainty"]
    print(f"    (Check: REL - RES + UNC = {check_val:.6f})")
    print(f"    NLL = {nll_post:.6f}")

    del y_val, y_prob_val_raw, y_prob_test_raw, y_prob_test_platt
    gc.collect()

    return {
        "task": task_id,
        "horizon": horizon,
        "n_test": int(len(y_test)),
        "prevalence": float(y_test.mean()),
        "pre_calibration": pre_decomp,
        "post_calibration": post_decomp,
        "post_nll": float(nll_post),
        "platt_params": {"A": float(A), "B": float(B)},
        "platt_status": status,
    }

def main():
    all_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "15_brier_decomposition.py",
            "latex_table": "Calibration Summary — extended with Murphy (1973) Brier decomposition",
            "binning": f"Q{ECE_N_BINS} (quantile, equal-frequency)",
            "reference": "Murphy (1973): Brier = Reliability - Resolution + Uncertainty",
        },
        "tasks": {},
    }

    for task_id in TASKS.keys():
        print_memory(f"before {task_id}")
        result = decompose_task(task_id)
        if result is not None:
            all_results["tasks"][task_id] = result
        gc.collect()
        print_memory(f"after {task_id}")

    # Save JSON
    output_path = RESULTS_DIR / "brier_decomposition.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")
    # LaTeX-ready rows
    print("\n" + "=" * 70)
    print("LATEX ROWS FOR CALIBRATION SUMMARY (post-calibration section)")
    tasks_order = ["T1", "T2", "T3", "T4"]

    def fmt(x, nd=6):
        return f"{x:.{nd}f}"

    rel_vals, res_vals, unc_vals, brier_vals, nll_vals = [], [], [], [], []
    for tid in tasks_order:
        if tid in all_results["tasks"]:
            post = all_results["tasks"][tid]["post_calibration"]
            rel_vals.append(fmt(post["reliability"], 6))
            res_vals.append(fmt(post["resolution"], 6))
            unc_vals.append(fmt(post["uncertainty"], 6))
            brier_vals.append(fmt(post["brier"], 6))
            nll_vals.append(f"{all_results['tasks'][tid]['post_nll']:.3f}")
        else:
            rel_vals.append("--")
            res_vals.append("--")
            unc_vals.append("--")
            brier_vals.append("--")
            nll_vals.append("--")

    print(f"Brier & {' & '.join(brier_vals)} \\\\")
    print(f"\\quad Reliability & {' & '.join(rel_vals)} \\\\")
    print(f"\\quad Resolution  & {' & '.join(res_vals)} \\\\")
    print(f"\\quad Uncertainty & {' & '.join(unc_vals)} \\\\")
    print(f"NLL & {' & '.join(nll_vals)} \\\\")
    print("\nNOTE: REL/RES are often tiny under rare events; printing 6 decimals avoids '0.0000'.")
    # Complete copy-paste table
    print("\n" + "=" * 70)
    print("COMPLETE CALIBRATION SUMMARY TABLE (copy-paste ready)")
    # Optionally load existing calibration.json for ECE/Slope values
    cal_path = RESULTS_DIR / "calibration.json"
    cal_data = {}
    if cal_path.exists():
        with open(cal_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f).get("tasks", {})

    print(r"\begin{table}[!t]")
    print(r"\caption{In-Domain Calibration Summary (Test Set). Murphy decomposition is used: $\text{Brier}=\text{Reliability}-\text{Resolution}+\text{Uncertainty}$.}")
    print(r"\label{tab:calibration_summary}")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Metric & T1 & T2 & T3 & T4 \\")
    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{Pre-calibration}} \\")

    # Pre-cal ECE
    ece_pre = []
    for tid in tasks_order:
        if tid in cal_data and "pre_calibration" in cal_data[tid] and "ece_q10" in cal_data[tid]["pre_calibration"]:
            ece_pre.append(f"{cal_data[tid]['pre_calibration']['ece_q10']:.3f}")
        else:
            ece_pre.append("--")
    print(f"ECE (Q10) & {' & '.join(ece_pre)} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{5}{l}{\textit{Post-calibration (Platt scaling)}} \\")

    # Post-cal ECE
    ece_post = []
    for tid in tasks_order:
        if tid in cal_data and "post_calibration" in cal_data[tid] and "ece_q10" in cal_data[tid]["post_calibration"]:
            ece_post.append(f"{cal_data[tid]['post_calibration']['ece_q10']:.4f}")
        else:
            ece_post.append("--")
    print(f"ECE (Q10) & {' & '.join(ece_post)} \\\\")

    # Brier from THIS decomposition (avoid mismatch with calibration.json)
    brier_post = []
    for tid in tasks_order:
        if tid in all_results["tasks"]:
            brier_post.append(f"{all_results['tasks'][tid]['post_calibration']['brier']:.4f}")
        else:
            brier_post.append("--")
    print(f"Brier & {' & '.join(brier_post)} \\\\")

    # Murphy components (from this script)
    print(f"\\quad Reliability & {' & '.join(rel_vals)} \\\\")
    print(f"\\quad Resolution  & {' & '.join(res_vals)} \\\\")
    print(f"\\quad Uncertainty & {' & '.join(unc_vals)} \\\\")

    # Slope from calibration.json if available
    slope_post = []
    for tid in tasks_order:
        if tid in cal_data and "post_calibration" in cal_data[tid] and "slope" in cal_data[tid]["post_calibration"]:
            slope_post.append(f"{cal_data[tid]['post_calibration']['slope']:.3f}")
        else:
            slope_post.append("--")
    print(f"Slope & {' & '.join(slope_post)} \\\\")

    # NLL from this script
    print(f"NLL & {' & '.join(nll_vals)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\nBRIER DECOMPOSITION COMPLETE")

if __name__ == "__main__":
    main()