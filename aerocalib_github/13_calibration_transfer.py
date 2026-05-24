#!/usr/bin/env python3
"""Measures calibration behavior under domain shift with systematic 2x2 ablation."""

import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")

from config import (
    PROCESSED_DIR, RESULTS_DIR, CLIMATE_ZONES,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight,
    get_feature_columns, ECE_N_BINS
)

np.random.seed(RANDOM_SEED)
EPS = 1e-10
HUGE = 1e15

# Fine-tuning adds this many boosting rounds (incremental)
FT_N_ESTIMATORS = 50

# Fraction grid
DEFAULT_TARGET_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0]

# Rolling-origin (recal-only) settings
ENABLE_ROLLING_ORIGIN = True
ROLLING_STEPS_WEEKS = [1, 2, 4, 8, 12, 24]  # cumulative weeks
ROLLING_MIN_SAMPLES = 200
ROLLING_MIN_POS = 1

# Low-prob ECE threshold
ENABLE_LOWPROB_ECE = True
LOWPROB_THRESHOLD = 0.05
LOWPROB_N_BINS = 10

# Reliability bin stats (uniform bins) for key fractions only (kept small to limit JSON size)
ENABLE_BIN_STATS = True
BIN_STATS_KEY_FRACTIONS = {0.0, 0.01, 0.05, 0.10}
BIN_STATS_METHODS = {"zero_shot", "recal_only", "ft_recal"}
BIN_STATS_N_BINS = 15

# Scratch baseline (target-only model trained from scratch on sampled adapt set) can be expensive
# Keep OFF by default for 30-pair full sweep.
ENABLE_SCRATCH_BASELINE = False

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    return obj

def _finite_list(xs):
    out = []
    for x in xs:
        if x is None:
            continue
        try:
            if np.isfinite(x):
                out.append(float(x))
        except Exception:
            continue
    return out

def mean_std_finite(xs):
    xs = _finite_list(xs)
    if not xs:
        return np.nan, np.nan
    return float(np.mean(xs)), float(np.std(xs))

def finite_mean(xs):
    m, _ = mean_std_finite(xs)
    return m
# CALIBRATION + METRICS

def compute_ece(y_true, y_prob, n_bins=10, strategy="quantile"):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_prob) == 0:
        return np.nan

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        if len(np.unique(y_prob)) < 2:
            bins = np.linspace(0, 1, n_bins + 1)
        else:
            try:
                bins = np.percentile(y_prob, quantiles * 100)
                bins = np.unique(bins)
            except Exception:
                bins = np.linspace(0, 1, n_bins + 1)
    else:
        bins = np.linspace(0, 1, n_bins + 1)

    if len(bins) < 2:
        return float(np.abs(y_true.mean() - y_prob.mean()))

    try:
        bin_indices = np.digitize(y_prob, bins[1:-1])
    except Exception:
        return np.nan

    ece = 0.0
    total_count = len(y_true)
    for i in range(len(bins) - 1):
        mask_bin = (bin_indices == i)
        count = int(mask_bin.sum())
        if count > 0:
            avg_conf = float(np.mean(y_prob[mask_bin]))
            avg_acc = float(np.mean(y_true[mask_bin]))
            ece += count * abs(avg_acc - avg_conf)

    return float(ece / max(total_count, 1))

def compute_lowprob_ece(y_true, y_prob, threshold=0.05, n_bins=10):
    """
    ECE restricted to low-probability region (p <= threshold).
    Bins are uniform in [0, threshold] for stability.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    mask = np.isfinite(y_true) & np.isfinite(y_prob) & (y_prob <= threshold)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_prob) == 0:
        return np.nan

    bins = np.linspace(0, threshold, n_bins + 1)
    try:
        bin_indices = np.digitize(y_prob, bins[1:-1])
    except Exception:
        return np.nan

    ece = 0.0
    total_count = len(y_true)
    for i in range(len(bins) - 1):
        mb = (bin_indices == i)
        c = int(mb.sum())
        if c > 0:
            avg_conf = float(np.mean(y_prob[mb]))
            avg_acc = float(np.mean(y_true[mb]))
            ece += c * abs(avg_acc - avg_conf)
    return float(ece / max(total_count, 1))

def compute_reliability_bins(y_true, y_prob, n_bins=15):
    """
    Uniform-bin reliability summary for plotting later (kept compact).
    Returns bin edges + per-bin counts, avg_conf, avg_acc.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_prob) == 0:
        return None

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])

    counts = []
    avg_conf = []
    avg_acc = []

    for i in range(n_bins):
        mb = (idx == i)
        c = int(mb.sum())
        counts.append(c)
        if c > 0:
            avg_conf.append(float(np.mean(y_prob[mb])))
            avg_acc.append(float(np.mean(y_true[mb])))
        else:
            avg_conf.append(None)
            avg_acc.append(None)

    return {
        "n_bins": int(n_bins),
        "bin_edges": bins.tolist(),
        "counts": counts,
        "avg_conf": avg_conf,
        "avg_acc": avg_acc,
    }

def compute_calibration_slope_intercept(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    y_prob = np.clip(y_prob, EPS, 1 - EPS)
    logits = logit(y_prob)

    valid = np.isfinite(logits)
    logits = logits[valid]
    y_true = y_true[valid]

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    try:
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        lr.fit(logits.reshape(-1, 1), y_true)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan

def platt_scaling_fit(y_val, scores_val, fallback=(1.0, 0.0), C=1.0):
    y_val = np.asarray(y_val)
    scores_val = np.asarray(scores_val)

    mask = np.isfinite(y_val) & np.isfinite(scores_val)
    y_val = y_val[mask]
    scores_val = scores_val[mask]

    if len(np.unique(y_val)) < 2:
        return fallback

    scores_val = np.clip(scores_val, EPS, 1 - EPS)
    logits_val = logit(scores_val)

    mask_logits = np.isfinite(logits_val)
    y_val = y_val[mask_logits]
    logits_val = logits_val[mask_logits]

    if len(np.unique(y_val)) < 2:
        return fallback

    try:
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
        lr.fit(logits_val.reshape(-1, 1), y_val)

        A = float(lr.coef_[0][0])
        B = float(lr.intercept_[0])

        if (not np.isfinite(A)) or (A <= 0) or (not np.isfinite(B)):
            return fallback

        return A, B
    except Exception:
        return fallback

def platt_scaling_apply(scores, A, B):
    scores = np.asarray(scores)
    nan_mask = ~np.isfinite(scores)

    scores_safe = np.where(nan_mask, 0.5, scores)
    scores_safe = np.clip(scores_safe, EPS, 1 - EPS)

    logits_ = logit(scores_safe)
    p_cal = expit(A * logits_ + B)

    p_cal = np.asarray(p_cal, dtype=np.float64)
    p_cal[nan_mask] = np.nan
    return p_cal

def _filter_finite_pairs(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    return y_true[mask], y_prob[mask]

def compute_all_metrics(y_true, y_prob, include_lowprob=False):
    """
    Compute pooled discrimination + calibration metrics robustly.
    ECE/Brier computed if finite probs exist; AUC/NLL/slope require both classes.
    """
    y_true_f, y_prob_f = _filter_finite_pairs(y_true, y_prob)
    if len(y_prob_f) == 0:
        return {
            "auc": np.nan,
            "ece": np.nan,
            "brier": np.nan,
            "nll": np.nan,
            "cal_slope": np.nan,
            "cal_intercept": np.nan,
            "ece_lowprob": np.nan if include_lowprob else None,
        }

    y_prob_clip = np.clip(y_prob_f, EPS, 1 - EPS)
    has_both = (len(np.unique(y_true_f)) > 1)

    out = {
        "auc": roc_auc_score(y_true_f, y_prob_f) if has_both else np.nan,
        "ece": compute_ece(y_true_f, y_prob_f, ECE_N_BINS, "quantile"),
        "brier": brier_score_loss(y_true_f, y_prob_f),
        "nll": log_loss(y_true_f, y_prob_clip) if has_both else np.nan,
    }

    slope, intercept = compute_calibration_slope_intercept(y_true_f, y_prob_f)
    out["cal_slope"] = slope
    out["cal_intercept"] = intercept

    if include_lowprob:
        out["ece_lowprob"] = compute_lowprob_ece(
            y_true_f, y_prob_f, threshold=LOWPROB_THRESHOLD, n_bins=LOWPROB_N_BINS
        )

    return out

def compute_per_airport_metrics(y_true, y_prob, airports, include_lowprob=False):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    airports = np.asarray(airports)

    uniq = np.unique(airports)
    rows = []
    for apt in uniq:
        m = (airports == apt)
        if int(m.sum()) < 30:
            continue

        yt, yp = y_true[m], y_prob[m]
        met = compute_all_metrics(yt, yp, include_lowprob=include_lowprob)

        yt_f, yp_f = _filter_finite_pairs(yt, yp)
        met["airport"] = apt
        met["n_samples"] = int(len(yt_f))
        met["n_positive"] = int(np.sum(yt_f)) if len(yt_f) else 0
        met["has_both_classes"] = (len(np.unique(yt_f)) > 1) if len(yt_f) else False
        rows.append(met)

    return rows

def aggregate_airport_metrics(airport_metrics):
    if not airport_metrics:
        return None

    dfm = pd.DataFrame(airport_metrics)

    result = {
        "n_airports_total": int(len(dfm)),
        "n_airports_with_positives": int(dfm["has_both_classes"].sum()),
        "macro_ece": float(dfm["ece"].mean()),
        "macro_brier": float(dfm["brier"].mean()),
        "std_ece": float(dfm["ece"].std()),
        "worst10pct_ece": float(dfm.nlargest(max(1, len(dfm) // 10), "ece")["ece"].mean()),
        "ece_q25": float(dfm["ece"].quantile(0.25)),
        "ece_q75": float(dfm["ece"].quantile(0.75)),
    }

    df_auc = dfm[dfm["has_both_classes"] & dfm["auc"].notna()]
    if len(df_auc) > 0:
        result["macro_auc"] = float(df_auc["auc"].mean())
        result["std_auc"] = float(df_auc["auc"].std())
        result["worst10pct_auc"] = float(df_auc.nsmallest(max(1, len(df_auc) // 10), "auc")["auc"].mean())
        result["auc_q25"] = float(df_auc["auc"].quantile(0.25))
        result["auc_q75"] = float(df_auc["auc"].quantile(0.75))
    else:
        result.update({"macro_auc": np.nan, "std_auc": np.nan, "worst10pct_auc": np.nan, "auc_q25": np.nan, "auc_q75": np.nan})

    df_slope = dfm[dfm["cal_slope"].notna()]
    if len(df_slope) > 0:
        result["macro_slope"] = float(df_slope["cal_slope"].mean())
        result["macro_intercept"] = float(df_slope["cal_intercept"].mean())

    # Optional low-prob ECE macro if present
    if "ece_lowprob" in dfm.columns and dfm["ece_lowprob"].notna().any():
        result["macro_ece_lowprob"] = float(dfm["ece_lowprob"].mean())

    return result

def compute_gap_recovery(current_value, baseline_value, oracle_value, metric_type="auc"):
    """
    Gap recovery (%) from baseline->oracle.
    For AUC-like (higher better): (current-baseline)/(oracle-baseline)
    For ECE-like (lower better): (baseline-current)/(baseline-oracle)
    """
    if current_value is None or baseline_value is None or oracle_value is None:
        return None
    if np.isnan(current_value) or np.isnan(baseline_value) or np.isnan(oracle_value):
        return None

    if metric_type in ["auc", "auc_like", "bss", "nll_skill"]:
        gap = oracle_value - baseline_value
        if abs(gap) < EPS:
            return None
        improvement = current_value - baseline_value
    else:
        gap = baseline_value - oracle_value
        if abs(gap) < EPS:
            return None
        improvement = baseline_value - current_value

    recovery = (improvement / gap) * 100.0
    return float(np.clip(recovery, -100, 200))

def brier_climatology(y_true):
    y = np.asarray(y_true)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan, np.nan
    p = float(np.mean(y))
    return p * (1.0 - p), p

def nll_climatology(y_true):
    y = np.asarray(y_true)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan, np.nan
    p = float(np.mean(y))
    p = float(np.clip(p, EPS, 1 - EPS))
    yhat = np.full_like(y, p, dtype=np.float64)
    return float(log_loss(y, yhat)), p

def brier_skill(brier, brier_climo):
    if brier is None or brier_climo is None:
        return None
    if not np.isfinite(brier) or not np.isfinite(brier_climo) or brier_climo <= EPS:
        return None
    return float(1.0 - (brier / brier_climo))

def nll_skill(nll, nll_climo):
    if nll is None or nll_climo is None:
        return None
    if not np.isfinite(nll) or not np.isfinite(nll_climo) or nll_climo <= EPS:
        return None
    return float(1.0 - (nll / nll_climo))

def load_data():
    parquet_path = PROCESSED_DIR / "aerocalib_hourly.parquet"
    df = pd.read_parquet(parquet_path)
    df["year"] = df["datetime"].dt.year
    return df

def sanitize_features_inplace(df, cols, huge=HUGE):
    X = df[cols].to_numpy(np.float64, copy=False)
    bad = ~np.isfinite(X) | (np.abs(X) > huge)
    if bad.any():
        X = X.copy()
        X[bad] = np.nan
        df.loc[:, cols] = X
    return df

def replace_inf_huge_with_nan(X, huge=HUGE):
    X = np.asarray(X, dtype=np.float64)
    bad = np.isinf(X) | (np.abs(X) > huge)
    n_bad = int(bad.sum())
    if n_bad:
        X = X.copy()
        X[bad] = np.nan
    return X, n_bad
FEATURE_DISTANCE_CACHE = {}      # key: (src_zone, tgt_zone, n_features) -> dict
ZONE_TRAIN_PREV_CACHE = {}       # key: (zone, label_col) -> float
ZONE_MONTHLY_PREV_CACHE = {}     # key: (zone, label_col) -> list[12]

def _zone_airports(zone):
    return CLIMATE_ZONES.get(zone, [])

def get_zone_train_prevalence(df, zone, label_col):
    key = (zone, label_col)
    if key in ZONE_TRAIN_PREV_CACHE:
        return ZONE_TRAIN_PREV_CACHE[key]

    z_airports = _zone_airports(zone)
    if not z_airports:
        ZONE_TRAIN_PREV_CACHE[key] = np.nan
        return np.nan

    sub = df[
        (df["year"] >= TRAIN_YEARS[0]) &
        (df["year"] <= TRAIN_YEARS[1]) &
        (df["airport"].isin(z_airports))
    ].dropna(subset=[label_col])

    prev = float(sub[label_col].mean()) if len(sub) else np.nan
    ZONE_TRAIN_PREV_CACHE[key] = prev
    return prev

def get_zone_monthly_prevalence(df, zone, label_col):
    key = (zone, label_col)
    if key in ZONE_MONTHLY_PREV_CACHE:
        return ZONE_MONTHLY_PREV_CACHE[key]

    z_airports = _zone_airports(zone)
    if not z_airports:
        vec = [None] * 12
        ZONE_MONTHLY_PREV_CACHE[key] = vec
        return vec

    sub = df[
        (df["year"] >= TRAIN_YEARS[0]) &
        (df["year"] <= TRAIN_YEARS[1]) &
        (df["airport"].isin(z_airports))
    ].dropna(subset=[label_col])

    if len(sub) == 0:
        vec = [None] * 12
        ZONE_MONTHLY_PREV_CACHE[key] = vec
        return vec

    m = sub.assign(month=sub["datetime"].dt.month).groupby("month")[label_col].mean()
    vec = []
    for mm in range(1, 13):
        v = m.get(mm, np.nan)
        vec.append(None if (v is None or not np.isfinite(v)) else float(v))

    ZONE_MONTHLY_PREV_CACHE[key] = vec
    return vec

def seasonality_mismatch_from_monthly(src_vec, tgt_vec):
    xs, ys = [], []
    for a, b in zip(src_vec, tgt_vec):
        if a is None or b is None:
            continue
        if np.isfinite(a) and np.isfinite(b):
            xs.append(float(a))
            ys.append(float(b))
    if len(xs) < 2:
        return np.nan

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    # Pearson correlation
    try:
        rho = float(np.corrcoef(xs, ys)[0, 1])
    except Exception:
        return np.nan
    if not np.isfinite(rho):
        return np.nan
    return float((1.0 - rho) / 2.0)

def compute_feature_distance_pair(df, source_zone, target_zone, feature_cols):
    """
    Quantile distance over features, after z-scoring by target-train stats.
    Task-independent. Cached per pair.
    """
    key = (source_zone, target_zone, int(len(feature_cols)))
    if key in FEATURE_DISTANCE_CACHE:
        return FEATURE_DISTANCE_CACHE[key]

    src_airports = _zone_airports(source_zone)
    tgt_airports = _zone_airports(target_zone)

    if not src_airports or not tgt_airports:
        out = {"feature_distance": np.nan, "n_features_used": 0, "quantiles": [0.1, 0.5, 0.9]}
        FEATURE_DISTANCE_CACHE[key] = out
        return out

    src = df[
        (df["year"] >= TRAIN_YEARS[0]) &
        (df["year"] <= TRAIN_YEARS[1]) &
        (df["airport"].isin(src_airports))
    ]
    tgt = df[
        (df["year"] >= TRAIN_YEARS[0]) &
        (df["year"] <= TRAIN_YEARS[1]) &
        (df["airport"].isin(tgt_airports))
    ]

    if len(src) == 0 or len(tgt) == 0:
        out = {"feature_distance": np.nan, "n_features_used": 0, "quantiles": [0.1, 0.5, 0.9]}
        FEATURE_DISTANCE_CACHE[key] = out
        return out

    # Subsample for speed (optional)
    max_rows = 200_000
    rng = np.random.default_rng(0)
    if len(src) > max_rows:
        src = src.iloc[rng.choice(len(src), size=max_rows, replace=False)]
    if len(tgt) > max_rows:
        tgt = tgt.iloc[rng.choice(len(tgt), size=max_rows, replace=False)]

    qs = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    dists = []
    used = 0

    for f in feature_cols:
        if f not in src.columns or f not in tgt.columns:
            continue

        xs = src[f].to_numpy(np.float64, copy=False)
        xt = tgt[f].to_numpy(np.float64, copy=False)

        xs = xs[np.isfinite(xs)]
        xt = xt[np.isfinite(xt)]

        if len(xs) < 100 or len(xt) < 100:
            continue

        mu = float(np.mean(xt))
        sd = float(np.std(xt))
        if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd < 1e-8:
            continue

        xs_z = (xs - mu) / sd
        xt_z = (xt - mu) / sd

        try:
            q_src = np.nanquantile(xs_z, qs)
            q_tgt = np.nanquantile(xt_z, qs)
        except Exception:
            continue

        if not (np.all(np.isfinite(q_src)) and np.all(np.isfinite(q_tgt))):
            continue

        d = float(np.mean(np.abs(q_src - q_tgt)))
        if np.isfinite(d):
            dists.append(d)
            used += 1

    feat_dist = float(np.mean(dists)) if dists else np.nan
    out = {"feature_distance": feat_dist, "n_features_used": int(used), "quantiles": qs.tolist()}
    FEATURE_DISTANCE_CACHE[key] = out
    return out
# 2x2 ABLATION

def run_ablation_experiment(
    source_model, A_source, B_source,
    X_adapt, y_adapt, adapt_airports,
    X_test, y_test, test_airports,
    base_params, seed=42,
    compute_per_airport=False,
    compute_bin_stats=False
):
    """
    2x2 ablation: {No FT, FT} x {Source Cal, Target Cal}

    Returns methods:
      - zero_shot: source model + source calibration
      - recal_only: source model + target recalibration
      - ft_only: fine-tuned model + source calibration
      - ft_recal: fine-tuned model + target recalibration
    """
    results = {}

    n_adapt = len(y_adapt)
    n_cal = max(20, int(n_adapt * 0.3))
    n_cal = min(n_cal, max(1, n_adapt - 1))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_adapt)
    idx_recal = perm[:n_cal]
    idx_finetune = perm[n_cal:]

    X_finetune = X_adapt[idx_finetune]
    y_finetune = y_adapt[idx_finetune]
    X_recal = X_adapt[idx_recal]
    y_recal = y_adapt[idx_recal]

    include_lowprob = bool(ENABLE_LOWPROB_ECE)

    # --- ZERO-SHOT ---
    y_prob_raw = source_model.predict_proba(X_test)[:, 1]
    y_prob_zeroshot = platt_scaling_apply(y_prob_raw, A_source, B_source)

    z_pooled = compute_all_metrics(y_test, y_prob_zeroshot, include_lowprob=include_lowprob)
    results["zero_shot"] = {"pooled": z_pooled}
    if compute_per_airport:
        results["zero_shot"]["per_airport"] = aggregate_airport_metrics(
            compute_per_airport_metrics(y_test, y_prob_zeroshot, test_airports, include_lowprob=include_lowprob)
        )
    if compute_bin_stats:
        results["zero_shot"]["reliability_bins"] = compute_reliability_bins(y_test, y_prob_zeroshot, n_bins=BIN_STATS_N_BINS)

    # --- RECAL-ONLY ---
    y_prob_recal_raw = source_model.predict_proba(X_recal)[:, 1]
    if len(np.unique(y_recal)) >= 2 and np.nanstd(y_prob_recal_raw) > EPS:
        A_target, B_target = platt_scaling_fit(y_recal, y_prob_recal_raw, fallback=(A_source, B_source))
    else:
        A_target, B_target = A_source, B_source

    y_prob_recal_only = platt_scaling_apply(y_prob_raw, A_target, B_target)
    r_pooled = compute_all_metrics(y_test, y_prob_recal_only, include_lowprob=include_lowprob)
    results["recal_only"] = {"pooled": r_pooled, "platt_params": {"A": A_target, "B": B_target}}
    if compute_per_airport:
        results["recal_only"]["per_airport"] = aggregate_airport_metrics(
            compute_per_airport_metrics(y_test, y_prob_recal_only, test_airports, include_lowprob=include_lowprob)
        )
    if compute_bin_stats:
        results["recal_only"]["reliability_bins"] = compute_reliability_bins(y_test, y_prob_recal_only, n_bins=BIN_STATS_N_BINS)

    # --- FT-ONLY and FT+RECAL ---
    if len(np.unique(y_finetune)) >= 2 and len(y_finetune) >= 50:
        ft_params = base_params.copy()
        ft_params["scale_pos_weight"] = compute_scale_pos_weight(float(np.mean(y_finetune)))
        ft_params["n_estimators"] = FT_N_ESTIMATORS

        ft_model = xgb.XGBClassifier(**ft_params, random_state=seed, use_label_encoder=False)
        ft_model.fit(X_finetune, y_finetune, xgb_model=source_model.get_booster(), verbose=False)

        y_prob_ft_raw = ft_model.predict_proba(X_test)[:, 1]

        y_prob_ft_only = platt_scaling_apply(y_prob_ft_raw, A_source, B_source)
        fto_pooled = compute_all_metrics(y_test, y_prob_ft_only, include_lowprob=include_lowprob)
        results["ft_only"] = {"pooled": fto_pooled}
        if compute_per_airport:
            results["ft_only"]["per_airport"] = aggregate_airport_metrics(
                compute_per_airport_metrics(y_test, y_prob_ft_only, test_airports, include_lowprob=include_lowprob)
            )
        if compute_bin_stats:
            results["ft_only"]["reliability_bins"] = compute_reliability_bins(y_test, y_prob_ft_only, n_bins=BIN_STATS_N_BINS)

        y_prob_ft_recal_raw = ft_model.predict_proba(X_recal)[:, 1]
        if len(np.unique(y_recal)) >= 2 and np.nanstd(y_prob_ft_recal_raw) > EPS:
            A_ft_target, B_ft_target = platt_scaling_fit(y_recal, y_prob_ft_recal_raw, fallback=(A_source, B_source))
        else:
            A_ft_target, B_ft_target = A_source, B_source

        y_prob_ft_recal = platt_scaling_apply(y_prob_ft_raw, A_ft_target, B_ft_target)
        ftr_pooled = compute_all_metrics(y_test, y_prob_ft_recal, include_lowprob=include_lowprob)
        results["ft_recal"] = {"pooled": ftr_pooled, "platt_params": {"A": A_ft_target, "B": B_ft_target}}
        if compute_per_airport:
            results["ft_recal"]["per_airport"] = aggregate_airport_metrics(
                compute_per_airport_metrics(y_test, y_prob_ft_recal, test_airports, include_lowprob=include_lowprob)
            )
        if compute_bin_stats:
            results["ft_recal"]["reliability_bins"] = compute_reliability_bins(y_test, y_prob_ft_recal, n_bins=BIN_STATS_N_BINS)
    else:
        results["ft_only"] = None
        results["ft_recal"] = None

    return results

def train_scratch_baseline(
    X_adapt, y_adapt,
    X_test, y_test,
    base_params, seed=42
):
    """
    Target-only scratch baseline on sampled adapt set.
    Uses a 70/30 split: 70% train, 30% calibrate (Platt).
    """
    n = len(y_adapt)
    if n < 200:
        return None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = max(50, int(n * 0.3))
    n_cal = min(n_cal, max(1, n - 1))

    idx_cal = perm[:n_cal]
    idx_tr = perm[n_cal:]

    X_tr, y_tr = X_adapt[idx_tr], y_adapt[idx_tr]
    X_cal, y_cal = X_adapt[idx_cal], y_adapt[idx_cal]

    if len(np.unique(y_tr)) < 2:
        return None

    p = base_params.copy()
    p["scale_pos_weight"] = compute_scale_pos_weight(float(np.mean(y_tr)))
    p.pop("early_stopping_rounds", None)
    p.pop("random_state", None)

    m = xgb.XGBClassifier(**p, random_state=seed, use_label_encoder=False)
    m.fit(X_tr, y_tr, verbose=False)

    s_cal = m.predict_proba(X_cal)[:, 1]
    A, B = platt_scaling_fit(y_cal, s_cal, fallback=(1.0, 0.0))

    s_test = m.predict_proba(X_test)[:, 1]
    p_test = platt_scaling_apply(s_test, A, B)

    met = compute_all_metrics(y_test, p_test, include_lowprob=bool(ENABLE_LOWPROB_ECE))
    met["platt_params"] = {"A": A, "B": B}
    return met
# ROLLING-ORIGIN RECAL-ONLY SIMULATION

def rolling_origin_recalibration(
    y_pool, s_pool, dt_pool,
    y_test, s_test,
    steps_weeks=None,
    min_samples=200,
    min_pos=1,
    fallback_platt=(1.0, 0.0)
):
    """
    Fit Platt scaling on increasing chronological windows of target_pool,
    then evaluate calibrated probabilities on target_test.

    Returns list of dicts sorted by weeks.
    """
    if steps_weeks is None:
        steps_weeks = [1, 2, 4, 8, 12, 24]

    # Ensure arrays
    y_pool = np.asarray(y_pool)
    s_pool = np.asarray(s_pool)
    dt_pool = pd.to_datetime(dt_pool)

    y_test = np.asarray(y_test)
    s_test = np.asarray(s_test)

    # Sort pool by datetime
    order = np.argsort(dt_pool.values)
    y_pool = y_pool[order]
    s_pool = s_pool[order]
    dt_pool = pd.Series(dt_pool).iloc[order].reset_index(drop=True)

    # Map each row to a "week bucket"
    week_period = dt_pool.dt.to_period("W")
    unique_weeks = pd.Index(sorted(week_period.unique()))

    out = []
    for k in steps_weeks:
        k = int(k)
        if k <= 0:
            continue

        if len(unique_weeks) == 0:
            break

        k_eff = min(k, len(unique_weeks))
        allowed_weeks = set(unique_weeks[:k_eff].tolist())
        mask = week_period.isin(allowed_weeks).to_numpy()

        y_fit = y_pool[mask]
        s_fit = s_pool[mask]

        # Filter finite
        mfit = np.isfinite(y_fit) & np.isfinite(s_fit)
        y_fit = y_fit[mfit]
        s_fit = s_fit[mfit]

        n_fit = int(len(y_fit))
        n_pos = int(np.sum(y_fit)) if n_fit else 0

        # Enforce minimums
        if n_fit < min_samples or n_pos < min_pos or len(np.unique(y_fit)) < 2:
            A, B = fallback_platt
        else:
            A, B = platt_scaling_fit(y_fit, s_fit, fallback=fallback_platt)

        p_test = platt_scaling_apply(s_test, A, B)
        met = compute_all_metrics(y_test, p_test, include_lowprob=bool(ENABLE_LOWPROB_ECE))

        out.append({
            "weeks": int(k_eff),
            "n_fit": int(n_fit),
            "n_pos": int(n_pos),
            "platt_params": {"A": A, "B": B},
            "metrics": met,
        })

    return out

def calibration_transfer_experiment(
    df, task_id, feature_cols,
    source_zone, target_zone,
    target_fractions, n_seeds=10
):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f"{task_id}_h{horizon}"

    print(f"\n{'=' * 70}")
    print(f"CALIBRATION TRANSFER: {source_zone} -> {target_zone}")
    print(f"Task: {task_id} ({task_config.name}), Horizon: {horizon}h")
    print(f"{'=' * 70}")

    source_airports = CLIMATE_ZONES.get(source_zone, [])
    target_airports = CLIMATE_ZONES.get(target_zone, [])

    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Features: {len(available_features)}")

    # sanitize once (idempotent)
    df = sanitize_features_inplace(df, available_features, huge=HUGE)
    tmp = df[available_features].to_numpy(np.float64, copy=False)
    print("INF count:", int(np.isinf(tmp).sum()))
    print("Huge count:", int((np.abs(tmp) > HUGE).sum()))
    # Prepare data splits
    source_train = df[
        (df["year"] >= TRAIN_YEARS[0]) &
        (df["year"] <= TRAIN_YEARS[1]) &
        (df["airport"].isin(source_airports))
    ].dropna(subset=[label_col])

    source_val = df[
        (df["year"] >= VAL_YEARS[0]) &
        (df["year"] <= VAL_YEARS[1]) &
        (df["airport"].isin(source_airports))
    ].dropna(subset=[label_col])

    source_test = df[
        (df["year"] >= TEST_YEARS[0]) &
        (df["year"] <= TEST_YEARS[1]) &
        (df["airport"].isin(source_airports))
    ].dropna(subset=[label_col])

    target_pool = df[
        (df["year"] >= TRAIN_YEARS[0]) &
        (df["year"] <= VAL_YEARS[1]) &
        (df["airport"].isin(target_airports))
    ].dropna(subset=[label_col])

    target_test = df[
        (df["year"] >= TEST_YEARS[0]) &
        (df["year"] <= TEST_YEARS[1]) &
        (df["airport"].isin(target_airports))
    ].dropna(subset=[label_col])

    print("\nData splits:")
    print(f"  Source train: {len(source_train):,}")
    print(f"  Source val:   {len(source_val):,}")
    print(f"  Source test:  {len(source_test):,}")
    print(f"  Target pool:  {len(target_pool):,}")
    print(f"  Target test:  {len(target_test):,}")

    if len(source_train) < 1000 or len(target_test) < 100:
        print("  ERROR: Insufficient data")
        return None

    # Arrays
    X_source_train = source_train[available_features].values
    y_source_train = source_train[label_col].values
    X_source_val = source_val[available_features].values
    y_source_val = source_val[label_col].values
    X_source_test = source_test[available_features].values
    y_source_test = source_test[label_col].values
    source_test_airports = source_test["airport"].values

    X_target_pool = target_pool[available_features].values
    y_target_pool = target_pool[label_col].values
    target_pool_airports = target_pool["airport"].values
    target_pool_dt = target_pool["datetime"].values

    X_target_test = target_test[available_features].values
    y_target_test = target_test[label_col].values
    target_test_airports = target_test["airport"].values

    # Replace only Inf/huge with NaN
    X_source_train, nrep = replace_inf_huge_with_nan(X_source_train, huge=HUGE)
    if nrep:
        print(f"[FIX] Replaced {nrep} (inf/huge) values in source_train with NaN")
    X_source_val, nrep = replace_inf_huge_with_nan(X_source_val, huge=HUGE)
    if nrep:
        print(f"[FIX] Replaced {nrep} (inf/huge) values in source_val with NaN")
    X_source_test, nrep = replace_inf_huge_with_nan(X_source_test, huge=HUGE)
    if nrep:
        print(f"[FIX] Replaced {nrep} (inf/huge) values in source_test with NaN")
    X_target_pool, nrep = replace_inf_huge_with_nan(X_target_pool, huge=HUGE)
    if nrep:
        print(f"[FIX] Replaced {nrep} (inf/huge) values in target_pool with NaN")
    X_target_test, nrep = replace_inf_huge_with_nan(X_target_test, huge=HUGE)
    if nrep:
        print(f"[FIX] Replaced {nrep} (inf/huge) values in target_test with NaN")
    # prevalence_ratio + seasonality_mismatch are task-dependent
    prev_src_train = get_zone_train_prevalence(df, source_zone, label_col)
    prev_tgt_train = get_zone_train_prevalence(df, target_zone, label_col)
    prev_ratio = (prev_tgt_train / (prev_src_train + EPS)) if np.isfinite(prev_src_train) and np.isfinite(prev_tgt_train) else np.nan

    src_month = get_zone_monthly_prevalence(df, source_zone, label_col)
    tgt_month = get_zone_monthly_prevalence(df, target_zone, label_col)
    season_mis = seasonality_mismatch_from_monthly(src_month, tgt_month)

    # feature_distance is task-independent, cached per pair
    fd = compute_feature_distance_pair(df, source_zone, target_zone, available_features)

    shift_summary = {
        "train_years": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
        "prevalence_ratio": float(prev_ratio) if np.isfinite(prev_ratio) else np.nan,
        "feature_distance": fd.get("feature_distance", np.nan),
        "feature_distance_n_features_used": fd.get("n_features_used", 0),
        "feature_distance_quantiles": fd.get("quantiles", [0.1, 0.5, 0.9]),
        "seasonality_mismatch": float(season_mis) if np.isfinite(season_mis) else np.nan,
    }
    print(f"\n[Shift summary TRAIN {shift_summary['train_years']}] "
          f"prev_ratio={shift_summary['prevalence_ratio']:.4f}, "
          f"feat_dist={shift_summary['feature_distance']:.4f}, "
          f"season_mis={shift_summary['seasonality_mismatch']:.4f}")
    # Step 1: Train source model
    print("\n[Step 1] Training source model...")

    scale_pos_weight = compute_scale_pos_weight(float(np.mean(y_source_train)))
    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    params.pop("early_stopping_rounds", None)
    params.pop("random_state", None)

    source_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    source_model.fit(X_source_train, y_source_train, verbose=False)
    # Step 2: Calibrate on source val
    print("[Step 2] Calibrating on source validation...")

    y_prob_source_val_raw = source_model.predict_proba(X_source_val)[:, 1]
    A_source, B_source = platt_scaling_fit(y_source_val, y_prob_source_val_raw, fallback=(1.0, 0.0))
    print(f"  Source Platt params: A={A_source:.4f}, B={B_source:.4f}")
    # Step 3: In-domain baseline (source test)
    print("\n[Step 3] In-domain evaluation...")

    y_prob_source_test_raw = source_model.predict_proba(X_source_test)[:, 1]
    y_prob_source_test_cal = platt_scaling_apply(y_prob_source_test_raw, A_source, B_source)

    source_metrics = {
        "pooled": compute_all_metrics(y_source_test, y_prob_source_test_cal, include_lowprob=bool(ENABLE_LOWPROB_ECE)),
        "per_airport": aggregate_airport_metrics(
            compute_per_airport_metrics(y_source_test, y_prob_source_test_cal, source_test_airports, include_lowprob=bool(ENABLE_LOWPROB_ECE))
        ),
    }

    print(
        f"  Pooled: AUC={source_metrics['pooled']['auc']:.4f}, "
        f"ECE={source_metrics['pooled']['ece']:.4f}, "
        f"slope={source_metrics['pooled']['cal_slope']:.3f}"
    )
    # Step 4: Zero-shot transfer
    print("\n[Step 4] Zero-shot transfer...")

    y_prob_target_raw = source_model.predict_proba(X_target_test)[:, 1]
    y_prob_target_zeroshot = platt_scaling_apply(y_prob_target_raw, A_source, B_source)

    zeroshot_metrics = {
        "pooled": compute_all_metrics(y_target_test, y_prob_target_zeroshot, include_lowprob=bool(ENABLE_LOWPROB_ECE)),
        "per_airport": aggregate_airport_metrics(
            compute_per_airport_metrics(y_target_test, y_prob_target_zeroshot, target_test_airports, include_lowprob=bool(ENABLE_LOWPROB_ECE))
        ),
    }

    print(
        f"  Pooled: AUC={zeroshot_metrics['pooled']['auc']:.4f}, "
        f"ECE={zeroshot_metrics['pooled']['ece']:.4f}, "
        f"slope={zeroshot_metrics['pooled']['cal_slope']:.3f}"
    )
    if zeroshot_metrics["per_airport"]:
        print(
            f"  Macro:  AUC={zeroshot_metrics['per_airport']['macro_auc']:.4f}, "
            f"ECE={zeroshot_metrics['per_airport']['macro_ece']:.4f}"
        )

    # Rolling-origin recal-only (optional, cheap)
    rolling = None
    if ENABLE_ROLLING_ORIGIN:
        # Use source model raw scores on target_pool + target_test
        s_pool = source_model.predict_proba(X_target_pool)[:, 1]
        s_test = y_prob_target_raw  # already computed
        rolling = rolling_origin_recalibration(
            y_pool=y_target_pool, s_pool=s_pool, dt_pool=target_pool_dt,
            y_test=y_target_test, s_test=s_test,
            steps_weeks=ROLLING_STEPS_WEEKS,
            min_samples=ROLLING_MIN_SAMPLES,
            min_pos=ROLLING_MIN_POS,
            fallback_platt=(A_source, B_source),
        )
    # Step 5: Oracle (target-trained)
    print("\n[Step 5] Training oracle...")

    n_target_pool = len(y_target_pool)
    n_oracle_val = int(n_target_pool * 0.15)

    np.random.seed(RANDOM_SEED)
    idx_shuffle = np.random.permutation(n_target_pool)

    X_oracle_train = X_target_pool[idx_shuffle[n_oracle_val:]]
    y_oracle_train = y_target_pool[idx_shuffle[n_oracle_val:]]
    X_oracle_val = X_target_pool[idx_shuffle[:n_oracle_val]]
    y_oracle_val = y_target_pool[idx_shuffle[:n_oracle_val]]

    oracle_params = params.copy()
    oracle_params["scale_pos_weight"] = compute_scale_pos_weight(float(np.mean(y_oracle_train)))

    oracle_model = xgb.XGBClassifier(**oracle_params, random_state=RANDOM_SEED, use_label_encoder=False)
    oracle_model.fit(X_oracle_train, y_oracle_train, verbose=False)

    y_prob_oracle_val_raw = oracle_model.predict_proba(X_oracle_val)[:, 1]
    A_oracle, B_oracle = platt_scaling_fit(y_oracle_val, y_prob_oracle_val_raw, fallback=(1.0, 0.0))

    y_prob_oracle_test_raw = oracle_model.predict_proba(X_target_test)[:, 1]
    y_prob_oracle_test_cal = platt_scaling_apply(y_prob_oracle_test_raw, A_oracle, B_oracle)

    oracle_metrics = {
        "pooled": compute_all_metrics(y_target_test, y_prob_oracle_test_cal, include_lowprob=bool(ENABLE_LOWPROB_ECE)),
        "per_airport": aggregate_airport_metrics(
            compute_per_airport_metrics(y_target_test, y_prob_oracle_test_cal, target_test_airports, include_lowprob=bool(ENABLE_LOWPROB_ECE))
        ),
    }

    print(
        f"  Pooled: AUC={oracle_metrics['pooled']['auc']:.4f}, "
        f"ECE={oracle_metrics['pooled']['ece']:.4f}"
    )

    # Precompute climatology baselines on target_test
    brier_climo, p_climo = brier_climatology(y_target_test)
    nll_climo, _ = nll_climatology(y_target_test)
    # Step 6: Few-shot adaptation
    print(f"\n[Step 6] Few-shot adaptation (binomial stratified subsampling, {n_seeds} seeds)...")

    results = {
        "task": task_id,
        "task_name": task_config.name,
        "horizon": horizon,
        "source_zone": source_zone,
        "target_zone": target_zone,
        "n_features": len(available_features),
        "sampling_method": "repeated_binomial_stratified_subsampling",
        "ft_n_estimators": FT_N_ESTIMATORS,
        "shift_summary": shift_summary,
        "data_sizes": {
            "source_train": int(len(source_train)),
            "source_val": int(len(source_val)),
            "source_test": int(len(source_test)),
            "target_pool": int(len(target_pool)),
            "target_test": int(len(target_test)),
        },
        "prevalence": {
            "source_train": float(np.mean(y_source_train)),
            "target_pool": float(np.mean(y_target_pool)),
            "target_test": float(np.mean(y_target_test)),
        },
        "climatology": {
            "target_test_prevalence": float(p_climo) if np.isfinite(p_climo) else np.nan,
            "brier_climo": float(brier_climo) if np.isfinite(brier_climo) else np.nan,
            "nll_climo": float(nll_climo) if np.isfinite(nll_climo) else np.nan,
        },
        "source_platt_params": {"A": A_source, "B": B_source},
        "source_in_domain": source_metrics,
        "zero_shot_transfer": zeroshot_metrics,
        "oracle": oracle_metrics,
        "rolling_origin_recal_only": rolling,
        "adaptation_results": [],
    }

    zs_auc = zeroshot_metrics["pooled"]["auc"]
    zs_ece = zeroshot_metrics["pooled"]["ece"]
    zs_nll = zeroshot_metrics["pooled"]["nll"]
    zs_brier = zeroshot_metrics["pooled"]["brier"]

    or_auc = oracle_metrics["pooled"]["auc"]
    or_ece = oracle_metrics["pooled"]["ece"]
    or_nll = oracle_metrics["pooled"]["nll"]
    or_brier = oracle_metrics["pooled"]["brier"]

    pos_idx_global = np.where(y_target_pool == 1)[0]
    neg_idx_global = np.where(y_target_pool == 0)[0]
    pos_rate = len(pos_idx_global) / (len(pos_idx_global) + len(neg_idx_global) + EPS)

    for frac in target_fractions:
        print(f"\n  === {frac * 100:.0f}% target data ===")

        if frac == 0.0:
            # Optional bin stats at 0% (already in zero_shot_transfer, but keep fraction record consistent)
            results["adaptation_results"].append({
                "target_fraction": 0.0,
                "target_fraction_pct": "0%",
                "n_samples_target": 0,
                "n_samples_effective_target": 0,
                "n_samples_actual_mean": 0,
                "n_positive_mean": 0,
                "ablation": {
                    "zero_shot": {
                        "auc_mean": zs_auc, "auc_std": 0.0,
                        "ece_mean": zs_ece, "ece_std": 0.0,
                        "brier_mean": zs_brier, "brier_std": 0.0,
                        "nll_mean": zs_nll, "nll_std": 0.0,
                        "bss_mean": brier_skill(zs_brier, brier_climo),
                        "nll_skill_mean": nll_skill(zs_nll, nll_climo),
                    }
                },
                "best_auc_method": "zero_shot",
                "best_ece_method": "zero_shot",
                "best_auc": zs_auc,
                "best_ece": zs_ece,
                "auc_gap_recovery": 0.0,
                "ece_gap_recovery": 0.0,
            })
            continue

        if frac >= 1.0:
            results["adaptation_results"].append({
                "target_fraction": 1.0,
                "target_fraction_pct": "100%",
                "n_samples_target": int(len(y_target_pool)),
                "n_samples_effective_target": int(len(y_target_pool)),
                "n_samples_actual_mean": int(len(y_target_pool)),
                "n_positive_mean": int(np.sum(y_target_pool)),
                "ablation": {
                    "oracle": {
                        "auc_mean": or_auc, "auc_std": 0.0,
                        "ece_mean": or_ece, "ece_std": 0.0,
                        "brier_mean": or_brier, "brier_std": 0.0,
                        "nll_mean": or_nll, "nll_std": 0.0,
                        "bss_mean": brier_skill(or_brier, brier_climo),
                        "nll_skill_mean": nll_skill(or_nll, nll_climo),
                    }
                },
                "best_auc_method": "oracle",
                "best_ece_method": "oracle",
                "best_auc": or_auc,
                "best_ece": or_ece,
                "auc_gap_recovery": 100.0,
                "ece_gap_recovery": 100.0,
            })
            continue

        # Intended sample count for this fraction (min), but cannot exceed pool size
        n_samples_target = max(100, int(len(X_target_pool) * frac))
        n_samples_eff = min(n_samples_target, len(y_target_pool))

        compute_detailed = float(frac) in BIN_STATS_KEY_FRACTIONS

        seed_results = {
            "zero_shot": {"auc": [], "ece": [], "brier": [], "nll": [], "slope": [], "bss": [], "nll_skill": [], "ece_lowprob": []},
            "recal_only": {"auc": [], "ece": [], "brier": [], "nll": [], "slope": [], "bss": [], "nll_skill": [], "ece_lowprob": []},
            "ft_only": {"auc": [], "ece": [], "brier": [], "nll": [], "slope": [], "bss": [], "nll_skill": [], "ece_lowprob": []},
            "ft_recal": {"auc": [], "ece": [], "brier": [], "nll": [], "slope": [], "bss": [], "nll_skill": [], "ece_lowprob": []},
        }

        n_positive_list = []
        actual_n_samples_list = []
        n_seeds_attempted = 0
        n_seeds_zero_pos = 0

        detailed_ablations = []  # store a few full dicts only at key fractions

        for seed in range(RANDOM_SEED, RANDOM_SEED + n_seeds):
            np.random.seed(seed)
            n_seeds_attempted += 1

            # Binomial stratified sampling
            n_pos_sample = np.random.binomial(n_samples_eff, pos_rate)
            n_pos_sample = min(n_pos_sample, len(pos_idx_global))

            n_neg_sample = n_samples_eff - n_pos_sample
            n_neg_sample = min(n_neg_sample, len(neg_idx_global))

            # Top-up if class exhaustion
            actual_total = n_pos_sample + n_neg_sample
            if actual_total < n_samples_eff:
                remaining = n_samples_eff - actual_total

                pos_cap = len(pos_idx_global) - n_pos_sample
                neg_cap = len(neg_idx_global) - n_neg_sample

                if neg_cap >= pos_cap:
                    add_neg = min(remaining, neg_cap)
                    n_neg_sample += add_neg
                    remaining -= add_neg
                    if remaining > 0:
                        add_pos = min(remaining, pos_cap)
                        n_pos_sample += add_pos
                        remaining -= add_pos
                else:
                    add_pos = min(remaining, pos_cap)
                    n_pos_sample += add_pos
                    remaining -= add_pos
                    if remaining > 0:
                        add_neg = min(remaining, neg_cap)
                        n_neg_sample += add_neg
                        remaining -= add_neg

            if n_pos_sample == 0:
                n_seeds_zero_pos += 1

            sampled_pos = np.random.choice(pos_idx_global, n_pos_sample, replace=False) if n_pos_sample > 0 else np.array([], dtype=int)
            sampled_neg = np.random.choice(neg_idx_global, n_neg_sample, replace=False) if n_neg_sample > 0 else np.array([], dtype=int)
            sampled_idx = np.concatenate([sampled_pos, sampled_neg])

            X_adapt = X_target_pool[sampled_idx]
            X_adapt, _ = replace_inf_huge_with_nan(X_adapt, huge=HUGE)
            y_adapt = y_target_pool[sampled_idx]
            adapt_airports = target_pool_airports[sampled_idx]

            n_positive_list.append(int(np.sum(y_adapt)))
            actual_n_samples_list.append(int(len(y_adapt)))

            try:
                ablation = run_ablation_experiment(
                    source_model, A_source, B_source,
                    X_adapt, y_adapt, adapt_airports,
                    X_target_test, y_target_test, target_test_airports,
                    params, seed,
                    compute_per_airport=(compute_detailed and True),
                    compute_bin_stats=(compute_detailed and ENABLE_BIN_STATS),
                )

                if compute_detailed:
                    detailed_ablations.append((seed, ablation))

                for method in ["zero_shot", "recal_only", "ft_only", "ft_recal"]:
                    if ablation.get(method) and ablation[method].get("pooled"):
                        m = ablation[method]["pooled"]
                        auc_v = m.get("auc")
                        ece_v = m.get("ece")
                        brier_v = m.get("brier")
                        nll_v = m.get("nll")
                        slope_v = m.get("cal_slope")
                        ece_lp = m.get("ece_lowprob") if ENABLE_LOWPROB_ECE else None

                        seed_results[method]["auc"].append(auc_v)
                        seed_results[method]["ece"].append(ece_v)
                        seed_results[method]["brier"].append(brier_v)
                        seed_results[method]["nll"].append(nll_v)
                        seed_results[method]["slope"].append(slope_v)
                        seed_results[method]["ece_lowprob"].append(ece_lp)

                        seed_results[method]["bss"].append(brier_skill(brier_v, brier_climo))
                        seed_results[method]["nll_skill"].append(nll_skill(nll_v, nll_climo))

            except Exception as e:
                print(f"    Seed {seed} failed: {e}")
                continue

        # Success count based on ECE availability
        n_successful = len(_finite_list(seed_results["recal_only"]["ece"]))

        # Basic stats over sampled positives/samples
        n_pos_f = _finite_list(n_positive_list)
        n_samp_f = _finite_list(actual_n_samples_list)

        frac_result = {
            "target_fraction": float(frac),
            "target_fraction_pct": f"{frac * 100:.0f}%",
            "n_samples_target": int(n_samples_target),
            "n_samples_effective_target": int(n_samples_eff),
            "n_samples_actual_mean": int(np.mean(n_samp_f)) if n_samp_f else 0,
            "n_samples_actual_min": int(np.min(n_samp_f)) if n_samp_f else 0,
            "n_samples_actual_max": int(np.max(n_samp_f)) if n_samp_f else 0,
            "n_positive_mean": int(np.mean(n_pos_f)) if n_pos_f else 0,
            "n_positive_min": int(np.min(n_pos_f)) if n_pos_f else 0,
            "n_positive_max": int(np.max(n_pos_f)) if n_pos_f else 0,
            "n_seeds_attempted": int(n_seeds_attempted),
            "n_seeds_successful": int(n_successful),
            "n_seeds_with_zero_positives": int(n_seeds_zero_pos),
            "ablation": {},
        }

        # Optionally: one scratch baseline per fraction (expensive if enabled in full sweep)
        scratch_baseline = None
        if ENABLE_SCRATCH_BASELINE:
            # Deterministic sample for scratch baseline (use RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            n_pos_sample = np.random.binomial(n_samples_eff, pos_rate)
            n_pos_sample = min(n_pos_sample, len(pos_idx_global))
            n_neg_sample = n_samples_eff - n_pos_sample
            n_neg_sample = min(n_neg_sample, len(neg_idx_global))

            sampled_pos = np.random.choice(pos_idx_global, n_pos_sample, replace=False) if n_pos_sample > 0 else np.array([], dtype=int)
            sampled_neg = np.random.choice(neg_idx_global, n_neg_sample, replace=False) if n_neg_sample > 0 else np.array([], dtype=int)
            sampled_idx = np.concatenate([sampled_pos, sampled_neg])

            X_adapt_sb = X_target_pool[sampled_idx]
            X_adapt_sb, _ = replace_inf_huge_with_nan(X_adapt_sb, huge=HUGE)
            y_adapt_sb = y_target_pool[sampled_idx]

            scratch_baseline = train_scratch_baseline(
                X_adapt_sb, y_adapt_sb,
                X_target_test, y_target_test,
                params, seed=RANDOM_SEED
            )
            if scratch_baseline:
                frac_result["scratch_baseline"] = scratch_baseline

        # Best selection by AUC and by ECE
        best_auc = -np.inf
        best_auc_method = "zero_shot"
        best_ece = np.inf
        best_ece_method = "zero_shot"

        for method in ["zero_shot", "recal_only", "ft_only", "ft_recal"]:
            ece_mean, ece_std = mean_std_finite(seed_results[method]["ece"])
            if np.isnan(ece_mean):
                continue

            auc_mean, auc_std = mean_std_finite(seed_results[method]["auc"])
            brier_mean, brier_std = mean_std_finite(seed_results[method]["brier"])
            nll_mean, nll_std = mean_std_finite(seed_results[method]["nll"])
            slope_mean, slope_std = mean_std_finite(seed_results[method]["slope"])
            bss_mean, bss_std = mean_std_finite(seed_results[method]["bss"])
            nlls_mean, nlls_std = mean_std_finite(seed_results[method]["nll_skill"])

            method_metrics = {
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "ece_mean": ece_mean,
                "ece_std": ece_std,
                "brier_mean": brier_mean,
                "brier_std": brier_std,
                "nll_mean": nll_mean,
                "nll_std": nll_std,
                "bss_mean": bss_mean,
                "bss_std": bss_std,
                "nll_skill_mean": nlls_mean,
                "nll_skill_std": nlls_std,
            }

            if np.isfinite(slope_mean):
                method_metrics["slope_mean"] = slope_mean
                method_metrics["slope_std"] = slope_std

            if ENABLE_LOWPROB_ECE:
                ece_lp_mean, ece_lp_std = mean_std_finite(seed_results[method]["ece_lowprob"])
                method_metrics["ece_lowprob_mean"] = ece_lp_mean
                method_metrics["ece_lowprob_std"] = ece_lp_std

            # Attach bin stats for key fractions/methods using representative seed later
            frac_result["ablation"][method] = method_metrics

            if np.isfinite(auc_mean) and auc_mean > best_auc:
                best_auc = auc_mean
                best_auc_method = method

            if np.isfinite(ece_mean) and ece_mean < best_ece:
                best_ece = ece_mean
                best_ece_method = method

        if best_auc == -np.inf:
            best_auc = np.nan
            best_auc_method = "none"
        if best_ece == np.inf:
            best_ece = np.nan
            best_ece_method = "none"

        frac_result["best_auc_method"] = best_auc_method
        frac_result["best_auc"] = best_auc
        frac_result["best_ece_method"] = best_ece_method
        frac_result["best_ece"] = best_ece

        frac_result["auc_gap_recovery"] = compute_gap_recovery(best_auc, zs_auc, or_auc, "auc")
        frac_result["ece_gap_recovery"] = compute_gap_recovery(best_ece, zs_ece, or_ece, "ece")

        # Representative seed at key fractions (median ECE on recal_only)
        if compute_detailed and detailed_ablations:
            ece_values = []
            for s, a in detailed_ablations:
                if a.get("recal_only") and a["recal_only"].get("pooled"):
                    e = a["recal_only"]["pooled"].get("ece")
                    if e is not None and np.isfinite(e):
                        ece_values.append((s, float(e)))
            if ece_values:
                ece_values.sort(key=lambda x: x[1])
                representative_seed = int(ece_values[len(ece_values) // 2][0])

                # Extract a compact detailed record for a single seed (keeps JSON size reasonable)
                for s, a in detailed_ablations:
                    if int(s) != representative_seed:
                        continue

                    rep = {"seed": representative_seed, "methods": {}}

                    for mname in BIN_STATS_METHODS:
                        if mname in a and a[mname] is not None:
                            rep["methods"][mname] = {
                                "pooled": a[mname].get("pooled"),
                                "per_airport": a[mname].get("per_airport"),
                                "reliability_bins": a[mname].get("reliability_bins") if ENABLE_BIN_STATS else None,
                            }

                    frac_result["representative_detail"] = rep
                    break

        # Logging
        print(f"    Seeds: {n_successful}/{n_seeds_attempted} successful")
        print(f"    Seeds with 0 positives (included): {n_seeds_zero_pos}")

        print("    Results (best for each metric):")
        for method, metrics in frac_result["ablation"].items():
            marker_auc = " <-AUC" if method == best_auc_method else ""
            marker_ece = " <-ECE" if method == best_ece_method else ""

            slope_str = ""
            if "slope_mean" in metrics and np.isfinite(metrics["slope_mean"]):
                slope_str = f", slope={metrics['slope_mean']:.3f}"

            auc_show = metrics["auc_mean"]
            auc_std_show = metrics["auc_std"]
            ece_show = metrics["ece_mean"]
            ece_std_show = metrics["ece_std"]

            auc_txt = "nan" if np.isnan(auc_show) else f"{auc_show:.4f}+-{auc_std_show:.4f}"
            ece_txt = "nan" if np.isnan(ece_show) else f"{ece_show:.4f}+-{ece_std_show:.4f}"

            print(f"      {method:12s}: AUC={auc_txt}{marker_auc}, ECE={ece_txt}{marker_ece}{slope_str}")

        if frac_result["auc_gap_recovery"] is not None and frac_result["ece_gap_recovery"] is not None:
            print(
                f"    Recovery: AUC={frac_result['auc_gap_recovery']:.1f}% ({best_auc_method}), "
                f"ECE={frac_result['ece_gap_recovery']:.1f}% ({best_ece_method})"
            )

        results["adaptation_results"].append(frac_result)

    return results

def compute_pair_level_shift_summary(all_experiments):
    """
    Pair-level summary: median over tasks for each (source_zone -> target_zone).
    """
    rows = []
    for exp in all_experiments:
        ss = exp.get("shift_summary", {})
        rows.append({
            "pair": f"{exp.get('source_zone')}→{exp.get('target_zone')}",
            "prevalence_ratio": ss.get("prevalence_ratio", np.nan),
            "feature_distance": ss.get("feature_distance", np.nan),
            "seasonality_mismatch": ss.get("seasonality_mismatch", np.nan),
        })
    if not rows:
        return None

    dfm = pd.DataFrame(rows)
    out = {}
    for pair, g in dfm.groupby("pair"):
        out[pair] = {
            "prev_ratio_median": float(np.nanmedian(g["prevalence_ratio"].values)),
            "feat_dist_median": float(np.nanmedian(g["feature_distance"].values)),
            "season_mis_median": float(np.nanmedian(g["seasonality_mismatch"].values)),
            "n_tasks": int(len(g)),
        }
    return out

def main():
    print("CALIBRATION TRANSFER EXPERIMENT (v3.2.1 + all pairs)")
    print("\nKey improvements:")
    print("  - FT/recal split is random 70/30 to preserve prevalence in the recalibration fit set")
    print("  - 0-positive seeds are included (recal falls back when fitting is impossible)")
    print("  - SHIFT SUMMARY: prevalence_ratio, feature_distance, seasonality_mismatch")
    print("  - Fraction-level NLL + skill metrics (BSS, NLL skill) vs target-test climatology")
    print("  - Low-prob ECE (p<=0.05) for rare-event operating region")
    print("  - Rolling-origin recal-only (weeks-to-calibration) on target pool")
    print("  - Reliability bins + per-airport details stored only for key fractions (kept compact)")

    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    feature_cols = get_feature_columns()
    print(f"Features: {len(feature_cols)}")

    target_fractions = DEFAULT_TARGET_FRACTIONS
    # TRANSFER PAIRS: ALL DIRECTIONAL PAIRS (30 if 6 zones exist)
    PREFERRED_ZONE_ORDER = [
        "Southern Hemisphere",
        "Tropical",
        "East Asian",
        "Northern Europe",
        "Continental",
        "Mediterranean",
    ]

    zones = [z for z in PREFERRED_ZONE_ORDER if z in CLIMATE_ZONES and len(CLIMATE_ZONES.get(z, [])) > 0]
    if len(zones) < 2:
        zones = sorted([z for z, apts in CLIMATE_ZONES.items() if isinstance(apts, (list, tuple)) and len(apts) > 0])

    transfer_pairs = [(src, tgt) for src in zones for tgt in zones if src != tgt]

    print(f"\n[CONFIG] Climate zones used ({len(zones)}): {zones}")
    print(f"[CONFIG] Transfer pairs to run ({len(transfer_pairs)}):")
    for (s, t) in transfer_pairs:
        print(f"  - {s} -> {t}")

    tasks_to_run = ["T1", "T2", "T3", "T4"]

    all_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "13_calibration_transfer_v3_2_1_all_pairs.py",
            "version": "v3.2.1+all_pairs",
            "zones_used": zones,
            "n_transfer_pairs": int(len(transfer_pairs)),
            "methodology_notes": {
                "platt_scaling": "Logistic regression on logit(P_raw): P_cal = sigmoid(A*logit(P_raw)+B)",
                "sampling": "Binomial stratified subsampling: n_pos ~ Binomial(n_samples, prevalence)",
                "adaptation_split": "Random 70/30 split of sampled target adaptation set into FT and recal subsets",
                "ft_trees": f"Fine-tuning adds {FT_N_ESTIMATORS} trees (incremental boosting) with recomputed scale_pos_weight",
                "metrics": "ECE/Brier always if finite; AUC/NLL/slope require both classes; aggregation is NaN-robust",
                "zero_positive_seeds": "Included; recal-only falls back to source calibrator when fitting impossible",
                "skill_metrics": "BSS=1 - (Brier/Brier_climo), NLL skill=1 - (NLL/NLL_climo), climo uses p=target_test prevalence",
                "lowprob_ece": f"ECE over p<= {LOWPROB_THRESHOLD} with {LOWPROB_N_BINS} uniform bins",
                "rolling_origin": f"Recal-only Platt fit on first K weeks of target_pool; steps={ROLLING_STEPS_WEEKS}",
                "feature_distance": "Mean abs diff of z-scored quantiles (0.1,0.5,0.9) across features; z-score by target-train stats",
            },
            "target_fractions": target_fractions,
            "n_seeds": 10,
            "ece_config": {"n_bins": ECE_N_BINS, "strategy": "quantile"},
            "bin_stats_config": {
                "enabled": bool(ENABLE_BIN_STATS),
                "key_fractions": sorted(list(BIN_STATS_KEY_FRACTIONS)),
                "methods": sorted(list(BIN_STATS_METHODS)),
                "n_bins": BIN_STATS_N_BINS,
            },
            "scratch_baseline_enabled": bool(ENABLE_SCRATCH_BASELINE),
        },
        "experiments": [],
        "pair_level_shift_summary": None,
    }

    for source_zone, target_zone in transfer_pairs:
        print(f"\n{'#' * 70}")
        print(f"# TRANSFER: {source_zone} -> {target_zone}")
        print(f"{'#' * 70}")

        for task_id in tasks_to_run:
            result = calibration_transfer_experiment(
                df, task_id, feature_cols,
                source_zone, target_zone,
                target_fractions, n_seeds=10
            )
            if result:
                all_results["experiments"].append(result)
                print(f"\nOK: {task_id} completed")

    # Pair-level shift summary (median over tasks)
    all_results["pair_level_shift_summary"] = compute_pair_level_shift_summary(all_results["experiments"])

    output_path = RESULTS_DIR / "calibration_transfer.json"
    with open(output_path, "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    print(f"\nSaved: {output_path}")

    print("\n" + "=" * 70)
    print("SHIFT SUMMARY (PAIR-LEVEL, MEDIAN OVER TASKS)")
    if all_results["pair_level_shift_summary"]:
        for pair, ss in all_results["pair_level_shift_summary"].items():
            print(
                f"{pair}: prev_ratio~{ss['prev_ratio_median']:.4f}, "
                f"feat_dist~{ss['feat_dist_median']:.4f}, "
                f"season_mis~{ss['season_mis_median']:.4f}"
            )

    print("\n" + "=" * 70)
    print("SUMMARY: 2x2 ABLATION AT 5% TARGET DATA (AUC/ECE/NLL)")
    for exp in all_results["experiments"]:
        for r in exp["adaptation_results"]:
            if float(r["target_fraction"]) == 0.05:
                print(f"\n{exp['task']} ({exp['source_zone']} -> {exp['target_zone']}):")
                print(f"  {'Method':<15} {'AUC':>10} {'ECE':>10} {'NLL':>10} {'BSS':>10}")
                print(f"  {'-' * 60}")
                for method, metrics in r["ablation"].items():
                    auc = metrics.get("auc_mean", np.nan)
                    ece = metrics.get("ece_mean", np.nan)
                    nll = metrics.get("nll_mean", np.nan)
                    bss = metrics.get("bss_mean", np.nan)
                    auc_txt = "nan" if np.isnan(auc) else f"{auc:.4f}"
                    ece_txt = "nan" if np.isnan(ece) else f"{ece:.4f}"
                    nll_txt = "nan" if np.isnan(nll) else f"{nll:.4f}"
                    bss_txt = "nan" if np.isnan(bss) else f"{bss:.4f}"
                    print(f"  {method:<15} {auc_txt:>10} {ece_txt:>10} {nll_txt:>10} {bss_txt:>10}")
                print(f"  {'-' * 60}")
                print(f"  Best AUC: {r['best_auc_method']}, Best ECE: {r['best_ece_method']}")
                print(f"  Seeds with 0 positives (included): {r.get('n_seeds_with_zero_positives', 0)}")
                break

if __name__ == "__main__":
    main()
