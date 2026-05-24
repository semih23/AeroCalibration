#!/usr/bin/env python3
"""Temperature scaling (1-parameter) vs Platt scaling (2-parameter) under transfer."""

import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
from config import (
    PROCESSED_DIR, RESULTS_DIR, CLIMATE_ZONES,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight,
    get_feature_columns,
)

EPS = 1e-10
HUGE = 1e15
N_SEEDS = 10
TARGET_FRAC = 0.05

# 3 target regimes (covering hard/medium/mild shifts)
# Each paired with a geographically distant source
TRANSFER_PAIRS = [
    ("Southern Hemisphere", "Tropical"),
    ("Northern Europe", "Mediterranean"),
    ("Continental", "East Asian"),
]
TASK_IDS = ["T1", "T2", "T3", "T4"]

def replace_inf_huge(X):
    X = np.asarray(X, dtype=np.float64)
    bad = np.isinf(X) | (np.abs(X) > HUGE)
    if bad.any():
        X = X.copy(); X[bad] = np.nan
    return X

def platt_fit_2param(y, scores, fallback=(1.0, 0.0)):
    """Standard 2-parameter Platt: P = sigmoid(A*logit(P_raw) + B)"""
    y = np.asarray(y); scores = np.asarray(scores)
    mask = np.isfinite(y) & np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if len(np.unique(y)) < 2: return fallback
    scores = np.clip(scores, EPS, 1 - EPS)
    lo = logit(scores)
    mask2 = np.isfinite(lo); y, lo = y[mask2], lo[mask2]
    if len(np.unique(y)) < 2: return fallback
    try:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(lo.reshape(-1, 1), y)
        A, B = float(lr.coef_[0][0]), float(lr.intercept_[0])
        if not np.isfinite(A) or A <= 0 or not np.isfinite(B): return fallback
        return A, B
    except: return fallback

def temp_scaling_fit(y, scores):
    """1-parameter temperature scaling: P = sigmoid(A*logit(P_raw)), B=0"""
    y = np.asarray(y); scores = np.asarray(scores)
    mask = np.isfinite(y) & np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if len(np.unique(y)) < 2: return 1.0
    scores = np.clip(scores, EPS, 1 - EPS)
    lo = logit(scores)
    mask2 = np.isfinite(lo); y, lo = y[mask2], lo[mask2]
    if len(np.unique(y)) < 2: return 1.0

    def neg_ll(A):
        if A <= 0: return 1e10
        p = expit(A * lo)
        p = np.clip(p, EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize_scalar(neg_ll, bounds=(0.01, 20.0), method='bounded')
    return float(result.x) if result.success else 1.0

def apply_cal(scores, A, B):
    scores = np.asarray(scores)
    nan_mask = ~np.isfinite(scores)
    s = np.where(nan_mask, 0.5, scores)
    s = np.clip(s, EPS, 1 - EPS)
    p = expit(A * logit(s) + B)
    p = np.asarray(p, dtype=np.float64); p[nan_mask] = np.nan
    return p

def compute_ece(y_true, y_prob, n_bins=10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_prob) == 0: return np.nan
    try: bins = np.unique(np.percentile(y_prob, np.linspace(0, 100, n_bins + 1)))
    except: bins = np.linspace(0, 1, n_bins + 1)
    if len(bins) < 2: return float(np.abs(y_true.mean() - y_prob.mean()))
    idx = np.digitize(y_prob, bins[1:-1])
    ece = sum(int((idx == i).sum()) * abs(y_true[idx == i].mean() - y_prob[idx == i].mean())
              for i in range(len(bins) - 1) if (idx == i).sum() > 0)
    return float(ece / len(y_true))

def compute_nll(y_true, y_prob):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_prob) == 0 or len(np.unique(y_true)) < 2: return np.nan
    y_prob = np.clip(y_prob, EPS, 1 - EPS)
    return float(log_loss(y_true, y_prob))

def main():
    print("  Temperature Scaling vs Platt Scaling Comparison")
    df = pd.read_parquet(PROCESSED_DIR / "aerocalib_hourly.parquet")
    df["year"] = df["datetime"].dt.year
    feature_cols = get_feature_columns()
    avail = [c for c in feature_cols if c in df.columns]
    print(f"Loaded {len(df):,} rows, {len(avail)} features\n")

    results = []
    platt_wins_ece = 0
    platt_wins_nll = 0
    total_comparisons = 0

    for src_zone, tgt_zone in TRANSFER_PAIRS:
        src_ap = CLIMATE_ZONES[src_zone]
        tgt_ap = CLIMATE_ZONES[tgt_zone]

        # Data splits
        src_train = df[(df.year >= TRAIN_YEARS[0]) & (df.year <= TRAIN_YEARS[1]) &
                       df.airport.isin(src_ap)]
        src_val = df[(df.year >= VAL_YEARS[0]) & (df.year <= VAL_YEARS[1]) &
                     df.airport.isin(src_ap)]
        tgt_pool = df[(df.year >= TRAIN_YEARS[0]) & (df.year <= VAL_YEARS[1]) &
                      df.airport.isin(tgt_ap)]
        tgt_test = df[(df.year >= TEST_YEARS[0]) & (df.year <= TEST_YEARS[1]) &
                      df.airport.isin(tgt_ap)]

        for task_id in TASK_IDS:
            label_col = f"{task_id}_h{TASKS[task_id].horizon}"

            st = src_train.dropna(subset=[label_col])
            sv = src_val.dropna(subset=[label_col])
            tp = tgt_pool.dropna(subset=[label_col])
            tt = tgt_test.dropna(subset=[label_col])

            if len(st) < 1000 or len(tt) < 100:
                continue

            X_st = replace_inf_huge(st[avail].values)
            y_st = st[label_col].values
            X_sv = replace_inf_huge(sv[avail].values)
            y_sv = sv[label_col].values
            X_tp = replace_inf_huge(tp[avail].values)
            y_tp = tp[label_col].values
            X_tt = replace_inf_huge(tt[avail].values)
            y_tt = tt[label_col].values

            # Train source model
            spw = compute_scale_pos_weight(float(y_st.mean()))
            params = XGBOOST_PARAMS.copy()
            params["scale_pos_weight"] = spw
            params.pop("early_stopping_rounds", None)
            params.pop("random_state", None)

            model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
            model.fit(X_st, y_st, verbose=False)

            # Source calibration
            s_sv = model.predict_proba(X_sv)[:, 1]
            A_src, B_src = platt_fit_2param(y_sv, s_sv)

            # Target test scores
            s_tt = model.predict_proba(X_tt)[:, 1]

            # Seeded recalibration comparison
            pos_idx = np.where(y_tp == 1)[0]
            neg_idx = np.where(y_tp == 0)[0]
            n_sample = max(100, int(len(y_tp) * TARGET_FRAC))
            pos_rate = len(pos_idx) / len(y_tp)

            platt_eces, temp_eces = [], []
            platt_nlls, temp_nlls = [], []

            for seed in range(RANDOM_SEED, RANDOM_SEED + N_SEEDS):
                np.random.seed(seed)
                n_pos = min(np.random.binomial(n_sample, pos_rate), len(pos_idx))
                n_neg = min(n_sample - n_pos, len(neg_idx))
                idx = np.concatenate([
                    np.random.choice(pos_idx, n_pos, replace=False) if n_pos > 0 else [],
                    np.random.choice(neg_idx, n_neg, replace=False) if n_neg > 0 else [],
                ]).astype(int)

                y_cal = y_tp[idx]
                s_cal = model.predict_proba(X_tp[idx])[:, 1]

                if len(np.unique(y_cal)) < 2:
                    continue

                # 2-param Platt
                A2, B2 = platt_fit_2param(y_cal, s_cal, fallback=(A_src, B_src))
                p2 = apply_cal(s_tt, A2, B2)

                # 1-param temp scaling
                A1 = temp_scaling_fit(y_cal, s_cal)
                p1 = apply_cal(s_tt, A1, 0.0)

                platt_eces.append(compute_ece(y_tt, p2))
                temp_eces.append(compute_ece(y_tt, p1))
                platt_nlls.append(compute_nll(y_tt, p2))
                temp_nlls.append(compute_nll(y_tt, p1))

            if not platt_eces:
                continue

            pe = np.nanmean(platt_eces)
            te = np.nanmean(temp_eces)
            pn = np.nanmean(platt_nlls)
            tn = np.nanmean(temp_nlls)

            if pe < te: platt_wins_ece += 1
            if pn < tn: platt_wins_nll += 1
            total_comparisons += 1

            delta_ece = te - pe  # positive = Platt better
            delta_nll = tn - pn

            print(f"{src_zone[:6]}→{tgt_zone[:6]} {task_id}: "
                  f"Platt ECE={pe:.4f} Temp ECE={te:.4f} (Δ={delta_ece:+.4f}) | "
                  f"Platt NLL={pn:.4f} Temp NLL={tn:.4f} (Δ={delta_nll:+.4f})")

            results.append({
                "source": src_zone, "target": tgt_zone, "task": task_id,
                "platt_ece": pe, "temp_ece": te, "delta_ece": delta_ece,
                "platt_nll": pn, "temp_nll": tn, "delta_nll": delta_nll,
            })

    # Summary
    print(f"\n{'='*65}")
    print(f"Platt wins ECE: {platt_wins_ece}/{total_comparisons} ({platt_wins_ece/total_comparisons*100:.0f}%)")
    print(f"Platt wins NLL: {platt_wins_nll}/{total_comparisons} ({platt_wins_nll/total_comparisons*100:.0f}%)")

    deltas_ece = [r["delta_ece"] for r in results if np.isfinite(r["delta_ece"])]
    deltas_nll = [r["delta_nll"] for r in results if np.isfinite(r["delta_nll"])]
    if deltas_ece:
        print(f"Median ΔECE (temp−platt): {np.median(deltas_ece):.4f}")
    if deltas_nll:
        print(f"Median ΔNLL (temp−platt): {np.median(deltas_nll):.4f}")

    print(f"\n=== PROSE FOR PAPER (replace temp scaling sentence in Section II-C) ===")
    print(f"Temperature scaling (one parameter, $B=0$) was also evaluated on "
          f"{total_comparisons} transfer-task combinations at 5\\% target data. "
          f"Platt scaling achieved lower ECE in {platt_wins_ece} of {total_comparisons} "
          f"cases and lower NLL in {platt_wins_nll} of {total_comparisons}; "
          f"the median ECE difference was {np.median(deltas_ece):.4f}.")

    out = {
        "metadata": {"generated_at": datetime.now().isoformat(), "n_seeds": N_SEEDS,
                      "target_fraction": TARGET_FRAC},
        "results": results,
        "summary": {
            "platt_wins_ece": platt_wins_ece, "platt_wins_nll": platt_wins_nll,
            "total": total_comparisons,
            "median_delta_ece": float(np.median(deltas_ece)) if deltas_ece else None,
            "median_delta_nll": float(np.median(deltas_nll)) if deltas_nll else None,
        }
    }
    with open(RESULTS_DIR / "temp_vs_platt.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: results/temp_vs_platt.json")

if __name__ == "__main__":
    main()
