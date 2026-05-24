#!/usr/bin/env python3
"""Rolling-origin Platt recalibration with cumulative target windows."""

import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
from config import (
    PROCESSED_DIR, RESULTS_DIR, CLIMATE_ZONES,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight,
    get_feature_columns, ECE_N_BINS,
)

# ── settings ──────────────────────────────────────────────────────────
EPS = 1e-10
HUGE = 1e15
FT_N_ESTIMATORS = 50
N_SEEDS = 10                         # same as main calibration_transfer
WEEK_STEPS = [2, 4, 8, 12, 24, 52]  # cumulative weeks of target data

TRANSFER_PAIRS = [
    ("Southern Hemisphere", "Tropical"),
    ("Northern Europe", "Mediterranean"),
]
TASK_IDS = ["T1", "T4"]

# ── helpers (copied from 13_calibration_transfer.py) ──────────────────

def replace_inf_huge(X):
    X = np.asarray(X, dtype=np.float64)
    bad = np.isinf(X) | (np.abs(X) > HUGE)
    if bad.any():
        X = X.copy()
        X[bad] = np.nan
    return X

def platt_fit(y, scores, fallback=(1.0, 0.0)):
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

def platt_apply(scores, A, B):
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
        m = idx == i
        c = int(m.sum())
        if c > 0:
            ece += c * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece / len(y_true))

def compute_slope(y_true, y_prob):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(np.unique(y_true)) < 2 or np.nanstd(y_prob) < EPS:
        return np.nan
    p = np.clip(y_prob, EPS, 1 - EPS)
    lo = logit(p)
    mask2 = np.isfinite(lo)
    y_true, lo = y_true[mask2], lo[mask2]
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=500)
        lr.fit(lo.reshape(-1, 1), y_true)
        return float(lr.coef_[0][0])
    except Exception:
        return np.nan

def safe_auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))

# ── main experiment ───────────────────────────────────────────────────

def run_chronological_experiment(df, task_id, source_zone, target_zone, feature_cols):
    task_cfg = TASKS[task_id]
    label_col = f"{task_id}_h{task_cfg.horizon}"
    avail = [c for c in feature_cols if c in df.columns]

    src_ap = CLIMATE_ZONES[source_zone]
    tgt_ap = CLIMATE_ZONES[target_zone]

    # ── data splits ──
    src_train = df[(df.year >= TRAIN_YEARS[0]) & (df.year <= TRAIN_YEARS[1]) &
                   df.airport.isin(src_ap)].dropna(subset=[label_col])
    src_val   = df[(df.year >= VAL_YEARS[0]) & (df.year <= VAL_YEARS[1]) &
                   df.airport.isin(src_ap)].dropna(subset=[label_col])

    tgt_pool  = df[(df.year >= TRAIN_YEARS[0]) & (df.year <= VAL_YEARS[1]) &
                   df.airport.isin(tgt_ap)].dropna(subset=[label_col]).sort_values("datetime")
    tgt_test  = df[(df.year >= TEST_YEARS[0]) & (df.year <= TEST_YEARS[1]) &
                   df.airport.isin(tgt_ap)].dropna(subset=[label_col])

    print(f"\n{'='*65}")
    print(f"  {source_zone} -> {target_zone} | {task_id} ({task_cfg.name})")
    print(f"  src_train={len(src_train):,}  src_val={len(src_val):,}")
    print(f"  tgt_pool={len(tgt_pool):,}   tgt_test={len(tgt_test):,}")
    print(f"{'='*65}")

    X_src_tr = replace_inf_huge(src_train[avail].values)
    y_src_tr = src_train[label_col].values
    X_src_vl = replace_inf_huge(src_val[avail].values)
    y_src_vl = src_val[label_col].values
    X_tgt_pool = replace_inf_huge(tgt_pool[avail].values)
    y_tgt_pool = tgt_pool[label_col].values
    dt_tgt_pool = pd.to_datetime(tgt_pool["datetime"].values)
    X_tgt_test = replace_inf_huge(tgt_test[avail].values)
    y_tgt_test = tgt_test[label_col].values

    # ── 1) source model ──
    spw = compute_scale_pos_weight(float(y_src_tr.mean()))
    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = spw
    params.pop("early_stopping_rounds", None)
    params.pop("random_state", None)

    src_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    src_model.fit(X_src_tr, y_src_tr, verbose=False)
    print("  Source model trained.")

    # ── 2) source calibration ──
    s_val_raw = src_model.predict_proba(X_src_vl)[:, 1]
    A_src, B_src = platt_fit(y_src_vl, s_val_raw)
    print(f"  Source Platt: A={A_src:.3f}, B={B_src:.3f}")

    # ── 3) zero-shot on target test ──
    s_test_raw = src_model.predict_proba(X_tgt_test)[:, 1]
    p_zs = platt_apply(s_test_raw, A_src, B_src)
    zs_auc = safe_auc(y_tgt_test, p_zs)
    zs_ece = compute_ece(y_tgt_test, p_zs)
    zs_slope = compute_slope(y_tgt_test, p_zs)
    print(f"  Zero-shot: AUC={zs_auc:.3f}, ECE={zs_ece:.4f}, slope={zs_slope:.2f}")

    # ── 4) oracle ──
    n_pool = len(y_tgt_pool)
    n_orc_val = int(n_pool * 0.15)
    np.random.seed(RANDOM_SEED)
    perm = np.random.permutation(n_pool)
    X_orc_tr = X_tgt_pool[perm[n_orc_val:]]
    y_orc_tr = y_tgt_pool[perm[n_orc_val:]]
    X_orc_vl = X_tgt_pool[perm[:n_orc_val]]
    y_orc_vl = y_tgt_pool[perm[:n_orc_val]]

    orc_params = params.copy()
    orc_params["scale_pos_weight"] = compute_scale_pos_weight(float(y_orc_tr.mean()))
    orc_model = xgb.XGBClassifier(**orc_params, random_state=RANDOM_SEED, use_label_encoder=False)
    orc_model.fit(X_orc_tr, y_orc_tr, verbose=False)
    s_orc_vl = orc_model.predict_proba(X_orc_vl)[:, 1]
    A_orc, B_orc = platt_fit(y_orc_vl, s_orc_vl)
    p_orc = platt_apply(orc_model.predict_proba(X_tgt_test)[:, 1], A_orc, B_orc)
    orc_auc = safe_auc(y_tgt_test, p_orc)
    orc_ece = compute_ece(y_tgt_test, p_orc)
    print(f"  Oracle:    AUC={orc_auc:.3f}, ECE={orc_ece:.4f}")

    # ── 5) raw scores on target pool (for recal) ──
    s_pool_raw = src_model.predict_proba(X_tgt_pool)[:, 1]

    # ── 6) chronological weeks ──
    order = np.argsort(dt_tgt_pool.values)
    y_pool_ord   = y_tgt_pool[order]
    s_pool_ord   = s_pool_raw[order]
    X_pool_ord   = X_tgt_pool[order]
    dt_pool_ord  = dt_tgt_pool[order]

    week_period = pd.Series(dt_pool_ord).dt.to_period("W")
    unique_weeks = sorted(week_period.unique())

    results_by_week = []

    for K in WEEK_STEPS:
        K_eff = min(K, len(unique_weeks))
        allowed = set(list(unique_weeks)[:K_eff])
        mask = np.array([w in allowed for w in week_period])

        y_win = y_pool_ord[mask]
        s_win = s_pool_ord[mask]
        X_win = X_pool_ord[mask]

        n_win = int(len(y_win))
        n_pos = int(y_win.sum())
        approx_days = K_eff * 7

        print(f"\n  Week {K_eff:>3d} ({approx_days:>4d}d): n={n_win:,}, pos={n_pos}")

        if n_win < 50 or n_pos < 1:
            print("    -> Skipped (insufficient data)")
            results_by_week.append({
                "weeks": K_eff, "approx_days": approx_days,
                "n_samples": n_win, "n_pos": n_pos,
                "recal_only": None, "ft_recal": None,
            })
            continue

        # ── recal-only (deterministic, no seed loop needed for Platt) ──
        A_tgt, B_tgt = platt_fit(y_win, s_win, fallback=(A_src, B_src))
        p_recal = platt_apply(s_test_raw, A_tgt, B_tgt)
        recal_auc  = safe_auc(y_tgt_test, p_recal)
        recal_ece  = compute_ece(y_tgt_test, p_recal)
        recal_slope = compute_slope(y_tgt_test, p_recal)

        # recovery
        r_ece_recal = ((zs_ece - recal_ece) / (zs_ece - orc_ece)) if abs(zs_ece - orc_ece) > 1e-8 else np.nan
        r_auc_recal = ((recal_auc - zs_auc) / (orc_auc - zs_auc)) if abs(orc_auc - zs_auc) > 1e-8 else np.nan

        print(f"    Recal:  ECE={recal_ece:.4f} (R={r_ece_recal:.0%}), "
              f"AUC={recal_auc:.3f} (R={r_auc_recal:.0%}), slope={recal_slope:.2f}")

        # ── FT + Recal (seeded, 70/30 split within chronological window) ──
        ft_aucs, ft_eces, ft_slopes = [], [], []

        for seed in range(RANDOM_SEED, RANDOM_SEED + N_SEEDS):
            rng = np.random.default_rng(seed)
            perm_w = rng.permutation(n_win)
            n_cal = max(20, int(n_win * 0.3))
            n_cal = min(n_cal, n_win - 1)
            idx_cal = perm_w[:n_cal]
            idx_ft  = perm_w[n_cal:]

            X_ft, y_ft = X_win[idx_ft], y_win[idx_ft]
            X_cl, y_cl = X_win[idx_cal], y_win[idx_cal]
            s_cl = s_win[idx_cal]   # source model scores for recal subset

            if len(np.unique(y_ft)) < 2 or len(y_ft) < 50:
                continue

            try:
                ft_params = params.copy()
                ft_params["scale_pos_weight"] = compute_scale_pos_weight(float(y_ft.mean()))
                ft_params["n_estimators"] = FT_N_ESTIMATORS
                ft_model = xgb.XGBClassifier(**ft_params, random_state=seed, use_label_encoder=False)
                ft_model.fit(X_ft, y_ft, xgb_model=src_model.get_booster(), verbose=False)

                # recalibrate on the 30% holdout using FT model scores
                s_cl_ft = ft_model.predict_proba(X_cl)[:, 1]
                if len(np.unique(y_cl)) >= 2:
                    A_ft, B_ft = platt_fit(y_cl, s_cl_ft, fallback=(A_src, B_src))
                else:
                    A_ft, B_ft = A_src, B_src

                s_test_ft = ft_model.predict_proba(X_tgt_test)[:, 1]
                p_ft = platt_apply(s_test_ft, A_ft, B_ft)

                ft_aucs.append(safe_auc(y_tgt_test, p_ft))
                ft_eces.append(compute_ece(y_tgt_test, p_ft))
                ft_slopes.append(compute_slope(y_tgt_test, p_ft))

                del ft_model
            except Exception as e:
                print(f"    Seed {seed} failed: {e}")

        if ft_aucs:
            ft_auc_m = float(np.nanmean(ft_aucs))
            ft_ece_m = float(np.nanmean(ft_eces))
            ft_slope_m = float(np.nanmean(ft_slopes))
            r_ece_ft = ((zs_ece - ft_ece_m) / (zs_ece - orc_ece)) if abs(zs_ece - orc_ece) > 1e-8 else np.nan
            r_auc_ft = ((ft_auc_m - zs_auc) / (orc_auc - zs_auc)) if abs(orc_auc - zs_auc) > 1e-8 else np.nan
            print(f"    FT+R:   ECE={ft_ece_m:.4f} (R={r_ece_ft:.0%}), "
                  f"AUC={ft_auc_m:.3f} (R={r_auc_ft:.0%}), slope={ft_slope_m:.2f}")
        else:
            ft_auc_m = ft_ece_m = ft_slope_m = r_ece_ft = r_auc_ft = np.nan
            print("    FT+R:   all seeds failed")

        results_by_week.append({
            "weeks": K_eff,
            "approx_days": approx_days,
            "n_samples": n_win,
            "n_pos": n_pos,
            "recal_only": {
                "ece": recal_ece, "auc": recal_auc, "slope": recal_slope,
                "R_ece": float(r_ece_recal) if np.isfinite(r_ece_recal) else None,
                "R_auc": float(r_auc_recal) if np.isfinite(r_auc_recal) else None,
            },
            "ft_recal": {
                "ece": ft_ece_m, "auc": ft_auc_m, "slope": ft_slope_m,
                "R_ece": float(r_ece_ft) if np.isfinite(r_ece_ft) else None,
                "R_auc": float(r_auc_ft) if np.isfinite(r_auc_ft) else None,
                "auc_std": float(np.nanstd(ft_aucs)) if ft_aucs else None,
                "ece_std": float(np.nanstd(ft_eces)) if ft_eces else None,
            },
        })

    return {
        "source_zone": source_zone,
        "target_zone": target_zone,
        "task": task_id,
        "zero_shot": {"auc": zs_auc, "ece": zs_ece, "slope": zs_slope},
        "oracle":    {"auc": orc_auc, "ece": orc_ece},
        "chronological": results_by_week,
    }

# ── main ──────────────────────────────────────────────────────────────

def main():
    print("  Chronological Adaptation Experiment")
    df = pd.read_parquet(PROCESSED_DIR / "aerocalib_hourly.parquet")
    df["year"] = df["datetime"].dt.year
    print(f"Loaded {len(df):,} rows")

    feature_cols = get_feature_columns()
    print(f"Features: {len(feature_cols)}")

    all_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "17_chronological_adaptation.py",
            "week_steps": WEEK_STEPS,
            "n_seeds": N_SEEDS,
            "description": "Genuine chronological accumulation (first K weeks of target pool)",
        },
        "experiments": [],
    }

    for src, tgt in TRANSFER_PAIRS:
        for tid in TASK_IDS:
            result = run_chronological_experiment(df, tid, src, tgt, feature_cols)
            if result:
                all_results["experiments"].append(result)

    # ── save ──
    out_path = RESULTS_DIR / "chronological_adaptation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # ── summary ──
    print("\n" + "=" * 65)
    print("  SUMMARY FOR LATEX TABLE")
    for exp in all_results["experiments"]:
        print(f"\n{exp['source_zone']} -> {exp['target_zone']} ({exp['task']})")
        print(f"  Zero-shot: AUC={exp['zero_shot']['auc']:.3f}, ECE={exp['zero_shot']['ece']:.4f}")
        print(f"  Oracle:    AUC={exp['oracle']['auc']:.3f}, ECE={exp['oracle']['ece']:.4f}")
        print(f"  {'Weeks':>5s} {'Days':>5s} {'N':>7s} {'Pos':>5s} | "
              f"{'Recal ECE':>9s} {'R_ECE':>6s} | "
              f"{'FT AUC':>7s} {'R_AUC':>6s}")
        print("  " + "-" * 65)
        for w in exp["chronological"]:
            if w["recal_only"] is None:
                continue
            r = w["recal_only"]
            f = w["ft_recal"]
            r_ece_s = f"{r['R_ece']:.0%}" if r['R_ece'] is not None else "---"
            f_auc_s = f"{f['auc']:.3f}" if np.isfinite(f['auc']) else "---"
            f_rauc_s = f"{f['R_auc']:.0%}" if f['R_auc'] is not None else "---"
            print(f"  {w['weeks']:>5d} {w['approx_days']:>5d} {w['n_samples']:>7,d} {w['n_pos']:>5d} | "
                  f"{r['ece']:.4f} {r_ece_s:>6s} | "
                  f"{f_auc_s:>7s} {f_rauc_s:>6s}")

if __name__ == "__main__":
    main()
