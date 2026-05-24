#!/usr/bin/env python3
"""Computes per-regime prevalence for each task on PRE-TEST years (2010–2022),."""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import gc

from config import (
    PROCESSED_DIR, RESULTS_DIR,
    TASKS, TEST_YEARS, CLIMATE_ZONES, AIRPORTS,
    get_feature_columns
)

# Pre-test years = train + val (avoid coupling descriptive stats to final test)
PRETEST_YEARS = (2010, 2022)

def main():
    # Load data — only columns we need
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'

    label_cols = []
    for tid, tcfg in TASKS.items():
        label_cols.append(f'{tid}_h{tcfg.horizon}')

    needed = label_cols + ['datetime', 'airport']
    # 'year' may or may not be stored; we derive it from datetime
    try:
        df = pd.read_parquet(parquet_path, columns=needed)
    except Exception:
        # if 'airport' col is named differently, try 'station'
        needed_alt = label_cols + ['datetime', 'station']
        df = pd.read_parquet(parquet_path, columns=needed_alt)
        df.rename(columns={'station': 'airport'}, inplace=True)

    df['year'] = df['datetime'].dt.year.astype('int16')

    pre_mask = (df['year'] >= PRETEST_YEARS[0]) & (df['year'] <= PRETEST_YEARS[1])
    pre_df = df[pre_mask].copy()
    del df
    gc.collect()

    print(f"Pre-test rows (2010–2022): {len(pre_df):,}")
    # Build airport → regime mapping
    airport_to_zone = {}
    for zone, airports in CLIMATE_ZONES.items():
        for ap in airports:
            airport_to_zone[ap] = zone

    pre_df['zone'] = pre_df['airport'].map(airport_to_zone)
    # Drop airports not in any zone (e.g., held-out airports not in CLIMATE_ZONES)
    pre_df = pre_df.dropna(subset=['zone'])
    # Compute prevalence per regime per task
    zones_ordered = [
        'Northern Europe', 'Mediterranean', 'Continental',
        'East Asian', 'Tropical', 'Southern Hemisphere'
    ]

    results = {}
    print(f"\n{'Regime':<25}", end="")
    for tid in TASKS:
        print(f"  {tid:>8}", end="")
    print(f"  {'N_pre':>10}")
    for zone in zones_ordered:
        zone_df = pre_df[pre_df['zone'] == zone]
        n_total = len(zone_df)
        zone_results = {'n_pretest': int(n_total)}

        print(f"{zone:<25}", end="")
        for tid, tcfg in TASKS.items():
            lcol = f'{tid}_h{tcfg.horizon}'
            valid = zone_df[lcol].dropna()
            n_valid = len(valid)
            n_pos = int(valid.sum())
            prev = n_pos / n_valid if n_valid > 0 else 0.0
            zone_results[tid] = {
                'n_valid': int(n_valid),
                'n_positive': n_pos,
                'prevalence': float(prev),
                'prevalence_pct': f"{prev*100:.1f}"
            }
            print(f"  {prev*100:>7.1f}%", end="")
        print(f"  {n_total:>10,}")
        results[zone] = zone_results
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[!t]")
    latex_lines.append(
        r"\caption{Per-regime hazard prevalence (\%, pre-test years 2010--2022) used as descriptive statistics for the operational groupings.}"
    )
    latex_lines.append(r"\label{tab:regime_stats}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Regime & T1 (\%) & T2 (\%) & T3 (\%) & T4 (\%) \\")
    latex_lines.append(r"\midrule")

    for zone in zones_ordered:
        zr = results[zone]
        vals = []
        for tid in TASKS:
            vals.append(zr[tid]['prevalence_pct'])

        # Shorten zone name for table
        short = zone.replace('Southern Hemisphere', 'S.\ Hemisphere')
        short = short.replace('Northern Europe', 'N.\ Europe')

        latex_lines.append(f"{short} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines)
    print(f"\n{'='*70}")
    print("LATEX TABLE:")
    print(latex_str)
    # Save
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '14_regime_prevalence.py',
            'latex_table': 'Table (Per-Regime Prevalence)',
            'pretest_years': list(PRETEST_YEARS),
            'note': 'Descriptive regime statistics computed on pre-test years only.'
        },
        'regimes': results,
        'latex': latex_str
    }

    output_path = RESULTS_DIR / 'regime_prevalence.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    del pre_df
    gc.collect()

if __name__ == "__main__":
    main()