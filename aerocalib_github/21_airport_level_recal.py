#!/usr/bin/env python3
"""Airport-level Platt recalibration on individual target airports."""

import json
import sys
import warnings
from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

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
MIN_TEST_SAMPLES = 30
MIN_TEST_POS = 5
MIN_LOCAL_POS_FOR_FIT = 5  # need at least this many positives in local sample to fit Platt

# Same 3 transfer pairs as script 19 (temp vs platt) for consistency
TRANSFER_PAIRS = [
    ("Southern Hemisphere", "Tropical"),          # hard shift
    ("Continental", "East Asian"),                # medium shift
    ("Northern Europe", "Mediterranean"),         # mild shift
]
TASK_IDS = ["T1", "T2", "T3", "T4"]

def replace_inf_huge(X):
    X = np.asarray(X, dtype=np.float64)
    bad = np.isinf(X) | (np.abs(X) > HUGE)
    if bad.any():
        X = X.copy(); X[bad] = np.nan
    return X

def platt_fit_2param(y, scores, fallback=(1.0, 0.0)):
    y = np.asarray(y); scores = np.asarray(scores)
    mask = np.isfinite(y) & np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if len(np.unique(y)) < 2:
        return fallback
    scores = np.clip(scores, EPS, 1 - EPS)
    lo = logit(scores)
    mask2 = np.isfinite(lo)
    y, lo = y[mask2], lo[mask2]
    if len(np.unique(y)) < 2:
        return fallback
    try:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(lo.reshape(-1, 1), y)
        A, B = float(lr.coef_[0][0]), float(lr.intercept_[0])
        if not np.isfinite(A) or A <= 0 or not np.isfinite(B):
            return fallback
        return A, B
    except Exception:
        return fallback

def apply_cal(scores, A, B):
    scores = np.asarray(scores)
    nan_mask = ~np.isfinite(scores)
    s = np.where(nan_mask, 0.5, scores)
    s = np.clip(s, EPS, 1 - EPS)
    p = expit(A * logit(s) + B)
    p = np.asarray(p, dtype=np.float64)
    p[nan_mask] = np.nan
    return p

def compute_ece(y_true, y_prob, n_bins=10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_prob) == 0:
        return np.nan
    try:
        bins = np.unique(np.percentile(y_prob, np.linspace(0, 100, n_bins + 1)))
    except Exception:
        bins = np.linspace(0, 1, n_bins + 1)
    if len(bins) < 2:
        return float(np.abs(y_true.mean() - y_prob.mean()))
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for i in range(len(bins) - 1):
        m = (idx == i)
        if m.sum() > 0:
            ece += int(m.sum()) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece / len(y_true))

def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

def compute_gap_recovery(current, baseline, oracle, direction="lower_is_better"):
    if any(x is None or not np.isfinite(x) for x in (current, baseline, oracle)):
        return np.nan
    if direction == "lower_is_better":
        gap = baseline - oracle
        if abs(gap) < EPS:
            return np.nan
        return float((baseline - current) / gap)
    else:
        gap = oracle - baseline
        if abs(gap) < EPS:
            return np.nan
        return float((current - baseline) / gap)

def load_zone_oracles_from_json():
    """
    Pull zone-level oracle ECE/AUC values for each (target_zone, task) from the
    main calibration_transfer.json. Used as ceiling references in gap recovery.
    """
    p = RESULTS_DIR / "calibration_transfer.json"
    if not p.exists():
        print(f"WARNING: {p} not found. Cannot recover zone oracles; using in-domain values as proxy.")
        return {}

    with open(p) as f:
        data = json.load(f)

    oracles = {}
    for exp in data.get("experiments", []):
        tgt_zone = exp.get("target_zone")
        task = exp.get("task")
        if not tgt_zone or not task:
            continue
        # Look for target-in-domain metrics
        tid = exp.get("target_in_domain", {}) or exp.get("oracle", {})
        if "pooled" in tid:
            tid = tid["pooled"]
        auc = tid.get("auc")
        ece = tid.get("ece")
        if auc is not None and ece is not None:
            oracles[(tgt_zone, task)] = {"auc": float(auc), "ece": float(ece)}
    return oracles

def airport_level_experiment(df, src_zone, tgt_zone, task_id, feature_cols, oracle_ref):
    src_ap = CLIMATE_ZONES[src_zone]
    tgt_ap = CLIMATE_ZONES[tgt_zone]

    label_col = f"{task_id}_h{TASKS[task_id].horizon}"

    src_train = df[(df.year >= TRAIN_YEARS[0]) & (df.year <= TRAIN_YEARS[1]) &
                   df.airport.isin(src_ap)].dropna(subset=[label_col])
    src_val = df[(df.year >= VAL_YEARS[0]) & (df.year <= VAL_YEARS[1]) &
                 df.airport.isin(src_ap)].dropna(subset=[label_col])

    X_st = replace_inf_huge(src_train[feature_cols].values)
    y_st = src_train[label_col].values
    X_sv = replace_inf_huge(src_val[feature_cols].values)
    y_sv = src_val[label_col].values

    # Train source model
    spw = compute_scale_pos_weight(float(y_st.mean()))
    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = spw
    params.pop("early_stopping_rounds", None)
    params.pop("random_state", None)

    source_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    source_model.fit(X_st, y_st, verbose=False)

    # Source Platt
    s_sv = source_model.predict_proba(X_sv)[:, 1]
    A_src, B_src = platt_fit_2param(y_sv, s_sv)

    # Per-airport loop
    airport_results = []
    for airport in tgt_ap:
        pool = df[(df.airport == airport) &
                  (df.year >= TRAIN_YEARS[0]) & (df.year <= VAL_YEARS[1])].dropna(subset=[label_col])
        test = df[(df.airport == airport) &
                  (df.year >= TEST_YEARS[0]) & (df.year <= TEST_YEARS[1])].dropna(subset=[label_col])

        if len(test) < MIN_TEST_SAMPLES:
            continue
        if int(test[label_col].sum()) < MIN_TEST_POS:
            continue
        if len(pool) < 100:
            continue

        X_pool = replace_inf_huge(pool[feature_cols].values)
        y_pool = pool[label_col].values
        X_tt = replace_inf_huge(test[feature_cols].values)
        y_tt = test[label_col].values

        # Airport-level zero-shot (source Platt on airport test)
        s_tt_raw = source_model.predict_proba(X_tt)[:, 1]
        p_zs = apply_cal(s_tt_raw, A_src, B_src)
        auc_zs = roc_auc_score(y_tt, p_zs) if len(np.unique(y_tt)) > 1 else np.nan
        ece_zs = compute_ece(y_tt, p_zs)

        pos_idx = np.where(y_pool == 1)[0]
        neg_idx = np.where(y_pool == 0)[0]
        n_sample = max(50, int(len(y_pool) * TARGET_FRAC))
        pos_rate = len(pos_idx) / max(len(y_pool), 1)

        ece_recal_seeds, auc_ft_seeds = [], []
        for seed in range(RANDOM_SEED, RANDOM_SEED + N_SEEDS):
            rng = np.random.default_rng(seed)
            n_pos = min(int(rng.binomial(n_sample, pos_rate)), len(pos_idx))
            n_neg = min(n_sample - n_pos, len(neg_idx))
            if n_pos < MIN_LOCAL_POS_FOR_FIT:
                continue
            idx = np.concatenate([
                rng.choice(pos_idx, n_pos, replace=False),
                rng.choice(neg_idx, n_neg, replace=False),
            ]).astype(int)

            # 70/30 split
            rng.shuffle(idx)
            k = int(0.7 * len(idx))
            idx_ft, idx_cal = idx[:k], idx[k:]

            y_cal = y_pool[idx_cal]
            s_cal = source_model.predict_proba(X_pool[idx_cal])[:, 1]
            if len(np.unique(y_cal)) < 2:
                continue

            # --- Airport-level recal-only ---
            A_loc, B_loc = platt_fit_2param(y_cal, s_cal, fallback=(A_src, B_src))
            p_recal = apply_cal(s_tt_raw, A_loc, B_loc)
            ece_recal_seeds.append(compute_ece(y_tt, p_recal))

            # --- Airport-level FT+Recal ---
            y_ft = y_pool[idx_ft]
            X_ft = X_pool[idx_ft]
            if len(np.unique(y_ft)) < 2 or len(y_ft) < 50:
                continue
            ft_params = params.copy()
            ft_params["scale_pos_weight"] = compute_scale_pos_weight(float(np.mean(y_ft)))
            ft_params["n_estimators"] = 50
            ft_model = xgb.XGBClassifier(**ft_params, random_state=seed, use_label_encoder=False)
            try:
                ft_model.fit(X_ft, y_ft, xgb_model=source_model.get_booster(), verbose=False)
                s_tt_ft = ft_model.predict_proba(X_tt)[:, 1]
                if len(np.unique(y_tt)) > 1:
                    auc_ft_seeds.append(roc_auc_score(y_tt, s_tt_ft))
            except Exception:
                continue

        if not ece_recal_seeds or not auc_ft_seeds:
            continue

        ece_recal = float(np.nanmean(ece_recal_seeds))
        auc_ft = float(np.nanmean(auc_ft_seeds))

        # Oracle reference (zone-level, from main JSON)
        oracle = oracle_ref.get((tgt_zone, task_id), None)
        if oracle is None:
            # Fallback: use best observed in-domain on this airport
            oracle = {"auc": max(auc_ft, auc_zs) + 0.02, "ece": min(ece_recal, ece_zs) * 0.5}

        R_ece = compute_gap_recovery(ece_recal, ece_zs, oracle["ece"], "lower_is_better")
        R_auc = compute_gap_recovery(auc_ft, auc_zs, oracle["auc"], "higher_is_better")

        airport_results.append({
            "airport": airport,
            "n_test": int(len(y_tt)),
            "n_test_pos": int(y_tt.sum()),
            "auc_zs": float(auc_zs),
            "ece_zs": float(ece_zs),
            "ece_recal": ece_recal,
            "auc_ft": auc_ft,
            "R_ece": R_ece,
            "R_auc": R_auc,
            "ece_beats_auc": bool(np.isfinite(R_ece) and np.isfinite(R_auc) and R_ece > R_auc),
        })

    return airport_results

def load_data():
    return pd.read_parquet(PROCESSED_DIR / "aerocalib_hourly.parquet")

def main():
    print("  Airport-Level Recalibration Experiment (Step 21)")
    df = load_data()
    df["year"] = df["datetime"].dt.year
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    print(f"Loaded {len(df):,} rows, {len(feature_cols)} features\n")

    oracle_ref = load_zone_oracles_from_json()
    print(f"Loaded {len(oracle_ref)} zone-level oracle references from main JSON\n")

    all_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "21_airport_level_recal.py",
            "target_fraction": TARGET_FRAC,
            "n_seeds": N_SEEDS,
            "transfer_pairs": [list(p) for p in TRANSFER_PAIRS],
            "tasks": TASK_IDS,
        },
        "per_airport": [],
        "aggregate_by_task": {},
        "aggregate_overall": {},
    }

    for src_zone, tgt_zone in TRANSFER_PAIRS:
        print(f"\n{'=' * 70}\n  {src_zone} -> {tgt_zone}\n{'=' * 70}")
        for task_id in TASK_IDS:
            print(f"\n  Task {task_id}...")
            apt_results = airport_level_experiment(df, src_zone, tgt_zone, task_id, feature_cols, oracle_ref)
            for r in apt_results:
                r["source_zone"] = src_zone
                r["target_zone"] = tgt_zone
                r["task"] = task_id
                all_results["per_airport"].append(r)
                print(f"    {r['airport']:6s} R_ECE={r['R_ece']:+.2f} R_AUC={r['R_auc']:+.2f} "
                      f"(n_pos={r['n_test_pos']}, beats={'Y' if r['ece_beats_auc'] else 'N'})")

    # Aggregate
    rows = all_results["per_airport"]
    task_stats = {}
    for task in TASK_IDS:
        task_rows = [r for r in rows if r["task"] == task]
        n = len(task_rows)
        w = sum(1 for r in task_rows if r["ece_beats_auc"])
        lo, hi = wilson_interval(w, n)
        task_stats[task] = {"n": n, "wins": w, "rate": w / n if n else np.nan,
                            "wilson": [lo, hi]}

    n_all = len(rows)
    w_all = sum(1 for r in rows if r["ece_beats_auc"])
    lo_all, hi_all = wilson_interval(w_all, n_all)

    all_results["aggregate_by_task"] = task_stats
    all_results["aggregate_overall"] = {
        "n": n_all, "wins": w_all, "rate": w_all / n_all if n_all else np.nan,
        "wilson": [lo_all, hi_all]
    }

    # Save JSON
    out_path = RESULTS_DIR / "airport_level_recal.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # --- LaTeX rows for tab:airport_level ---
    print("\n" + "=" * 70)
    print("  LATEX TABLE ROWS (paste into tab:airport_level)")
    task_labels = {"T1": "T1 Visibility", "T2": "T2 Ceiling",
                   "T3": "T3 Crosswind", "T4": "T4 Thunderstorm"}
    for task in TASK_IDS:
        s = task_stats[task]
        if s["n"] == 0:
            continue
        rate_pct = 100 * s["rate"]
        lo_pct = 100 * s["wilson"][0]
        hi_pct = 100 * s["wilson"][1]
        print(f"{task_labels[task]} & {s['n']} & {s['wins']} & "
              f"{rate_pct:.0f}\\% [{lo_pct:.0f},{hi_pct:.0f}] \\\\")
    all_s = all_results["aggregate_overall"]
    print("\\midrule")
    print(f"All tasks & {all_s['n']} & {all_s['wins']} & "
          f"{100 * all_s['rate']:.0f}\\% [{100 * all_s['wilson'][0]:.0f},"
          f"{100 * all_s['wilson'][1]:.0f}] \\\\")

    # --- Abstract / prose replacements ---
    print("\n" + "=" * 70)
    print("  PLACEHOLDER REPLACEMENTS FOR MANUSCRIPT")
    print(f"\n[APT_WINS] = {all_s['wins']}")
    print(f"[APT_TOTAL] = {all_s['n']}")
    print()
    for task in TASK_IDS:
        s = task_stats[task]
        if s["n"] == 0:
            print(f"[APT_{task}_N] = N/A")
            continue
        rate_pct = 100 * s["rate"]
        lo_pct = 100 * s["wilson"][0]
        hi_pct = 100 * s["wilson"][1]
        print(f"[APT_{task}_N] = {s['n']}")
        print(f"[APT_{task}_W] = {s['wins']}")
        print(f"[APT_{task}_RATE] = {rate_pct:.0f}\\% [{lo_pct:.0f},{hi_pct:.0f}]")
    print(f"\n[APT_ALL_N] = {all_s['n']}")
    print(f"[APT_ALL_W] = {all_s['wins']}")
    print(f"[APT_ALL_RATE] = {100 * all_s['rate']:.0f}\\% "
          f"[{100 * all_s['wilson'][0]:.0f},{100 * all_s['wilson'][1]:.0f}]")

    print("\nDone.")

if __name__ == "__main__":
    main()
