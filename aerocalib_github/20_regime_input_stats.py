#!/usr/bin/env python3
"""Per-regime input feature distribution summaries."""

import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from config import (
    PROCESSED_DIR, RESULTS_DIR, CLIMATE_ZONES,
    TRAIN_YEARS, VAL_YEARS,
)

# Variables to summarize (raw METAR fields, lag-1 versions)
SUMMARY_VARS = {
    "visibility_lag1":  {"label": "Vis (m)",    "fmt": "{:.0f}"},
    "wind_speed_lag1":  {"label": "Wind (kt)",  "fmt": "{:.1f}"},
    "temperature_lag1": {"label": "Temp (°C)",  "fmt": "{:.1f}"},
}

# Missingness: fraction of NaN in key raw columns
MISS_VARS = ["visibility_lag1", "wind_speed_lag1", "temperature_lag1",
             "dewpoint_lag1"]

ZONE_ORDER = [
    "Northern Europe", "Mediterranean", "Continental",
    "East Asian", "Tropical", "Southern Hemisphere",
]
ZONE_SHORT = {
    "Northern Europe": "N.\\ Europe",
    "Mediterranean": "Mediterranean",
    "Continental": "Continental",
    "East Asian": "East Asian",
    "Tropical": "Tropical",
    "Southern Hemisphere": "S.\\ Hemisphere",
}

def main():
    print("  Input-Space Regime Statistics")
    df = pd.read_parquet(PROCESSED_DIR / "aerocalib_hourly.parquet")
    df["year"] = df["datetime"].dt.year

    # Use pre-test years only (consistent with regime_stats table)
    df = df[(df.year >= TRAIN_YEARS[0]) & (df.year <= VAL_YEARS[1])]
    print(f"Pre-test data: {len(df):,} rows\n")

    results = {}

    for zone in ZONE_ORDER:
        airports = CLIMATE_ZONES.get(zone, [])
        sub = df[df.airport.isin(airports)]
        n = len(sub)

        stats = {"n_records": n, "n_airports": len(airports)}

        # Summary statistics
        for var, cfg in SUMMARY_VARS.items():
            if var in sub.columns:
                vals = sub[var].dropna()
                stats[var] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(vals.median()),
                    "miss_pct": float(sub[var].isna().mean() * 100),
                }
            else:
                stats[var] = None

        # Overall missingness (average across key variables)
        miss_rates = []
        for var in MISS_VARS:
            if var in sub.columns:
                miss_rates.append(float(sub[var].isna().mean() * 100))
        stats["avg_miss_pct"] = float(np.mean(miss_rates)) if miss_rates else None

        results[zone] = stats

        # Print summary
        vis = stats.get("visibility_lag1", {})
        wind = stats.get("wind_speed_lag1", {})
        temp = stats.get("temperature_lag1", {})
        print(f"{zone:25s}  n={n:>8,d}  "
              f"Vis={vis.get('mean',0):>7.0f}±{vis.get('std',0):>5.0f}  "
              f"Wind={wind.get('mean',0):>4.1f}±{wind.get('std',0):>3.1f}  "
              f"Temp={temp.get('mean',0):>5.1f}±{temp.get('std',0):>4.1f}  "
              f"Miss={stats.get('avg_miss_pct',0):>4.1f}%")

    # ── LaTeX columns ──
    print(f"\n{'='*65}")
    print("OPTION A: Replace existing regime_stats table with wider version")
    print("(adds 3 columns: Vis μ, Wind μ, Miss%)")
    print(f"{'='*65}\n")

    print(r"\begin{table}[!t]")
    print(r"\caption{Per-regime hazard prevalence (\%, pre-test years 2010--2022) and input-space summary statistics.}")
    print(r"\label{tab:regime_stats}")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{2.5pt}")
    print(r"\begin{tabular}{lcccc|ccc}")
    print(r"\toprule")
    print(r"Regime & T1 & T2 & T3 & T4 & \shortstack{Vis\\(km)} & \shortstack{Wind\\(kt)} & \shortstack{Miss\\(\%)} \\")
    print(r"\midrule")

    # Prevalence values from the paper
    prev = {
        "Northern Europe":     ("1.4", "3.0", "2.7", "0.2"),
        "Mediterranean":       ("0.2", "1.0", "3.3", "0.4"),
        "Continental":         ("1.0", "1.2", "4.9", "0.5"),
        "East Asian":          ("0.5", "3.2", "2.2", "0.4"),
        "Tropical":            ("2.0", "4.6", "0.5", "2.1"),
        "Southern Hemisphere": ("0.8", "1.6", "2.3", "1.0"),
    }

    for zone in ZONE_ORDER:
        s = results[zone]
        vis_km = s["visibility_lag1"]["mean"] / 1000 if s.get("visibility_lag1") else 0
        wind_m = s["wind_speed_lag1"]["mean"] if s.get("wind_speed_lag1") else 0
        miss = s.get("avg_miss_pct", 0)
        p = prev[zone]
        short = ZONE_SHORT[zone]
        print(f"{short} & {p[0]} & {p[1]} & {p[2]} & {p[3]} "
              f"& {vis_km:.1f} & {wind_m:.1f} & {miss:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ── Save JSON ──
    out = {
        "metadata": {"generated_at": datetime.now().isoformat(),
                      "years": f"{TRAIN_YEARS[0]}-{VAL_YEARS[1]}"},
        "regimes": results,
    }
    with open(RESULTS_DIR / "regime_input_stats.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: results/regime_input_stats.json")

if __name__ == "__main__":
    main()
