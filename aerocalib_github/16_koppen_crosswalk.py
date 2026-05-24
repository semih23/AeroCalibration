#!/usr/bin/env python3
"""Builds an approximate crosswalk between your operational regimes and."""

import argparse
import gc
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR, RESULTS_DIR,
    CLIMATE_ZONES
)

DEFAULT_KG_URL = "https://koeppen-geiger.vu-wien.ac.at/data/Koeppen-Geiger-ASCII.txt"

def download_if_missing(url: str, path):
    if path.exists():
        return
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests is required to download KG ASCII. Install: pip install requests")

    print(f"Downloading Köppen–Geiger ASCII from:\n  {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"Saved KG ASCII to:\n  {path}")

def load_koppen_ascii(path):
    """
    Expects 2 header lines, then whitespace-separated columns:
    lat lon class
    """
    # Read as whitespace-delimited; skip first 2 header lines
    df = pd.read_csv(
        path,
        skiprows=2,
        header=None,
        delim_whitespace=True,
        names=["lat", "lon", "kg"],
        dtype={"lat": float, "lon": float, "kg": str},
        engine="python",
    )

    # Normalize lon to [-180, 180] if needed
    if df["lon"].max() > 180.0:
        df["lon"] = df["lon"].where(df["lon"] <= 180.0, df["lon"] - 360.0)

    df["kg"] = df["kg"].str.strip()
    return df

def extract_airport_coords_from_parquet(parquet_path):
    """
    Tries common latitude/longitude column names in your processed parquet.
    Returns DataFrame: airport, lat, lon (one row per airport).
    """
    # Candidate column name pairs (lat, lon)
    candidates = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("Latitude", "Longitude"),
        ("LAT", "LON"),
    ]

    # Load airport column name robustly
    base_cols = ["airport"]
    try:
        # quick probe
        pd.read_parquet(parquet_path, columns=["airport"], engine="auto")
        airport_col = "airport"
    except Exception:
        airport_col = "station"

    for lat_col, lon_col in candidates:
        try:
            df = pd.read_parquet(parquet_path, columns=[airport_col, lat_col, lon_col])
            df = df.rename(columns={airport_col: "airport", lat_col: "lat", lon_col: "lon"})
            df = df.dropna(subset=["lat", "lon"])
            coords = df.groupby("airport", as_index=False)[["lat", "lon"]].first()
            if len(coords) > 0:
                # Normalize lon
                coords["lon"] = coords["lon"].where(coords["lon"] <= 180.0, coords["lon"] - 360.0)
                return coords
        except Exception:
            continue

    raise RuntimeError(
        "Could not find latitude/longitude columns in aerocalib_hourly.parquet. "
        "Expected one of: (latitude,longitude), (lat,lon), etc."
    )

def nearest_koppen_class(kg_df, lat, lon):
    """
    Return KG class of nearest (lat,lon) in the ASCII grid.
    """
    dlat = kg_df["lat"].values - lat
    dlon = kg_df["lon"].values - lon
    dist2 = dlat * dlat + dlon * dlon
    idx = int(np.argmin(dist2))
    return kg_df["kg"].iloc[idx]

def format_counts(counter: Counter, max_items=6):
    """
    Format as: Cfb (4), Csa (2), ...
    If more than max_items, keep top max_items and group remainder as 'other (n)'.
    """
    items = counter.most_common()
    if not items:
        return "--"

    shown = items[:max_items]
    rest = items[max_items:]

    parts = [f"{k} ({v})" for k, v in shown]
    if rest:
        other_n = sum(v for _, v in rest)
        parts.append(f"other ({other_n})")
    return ", ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-url", default=DEFAULT_KG_URL, help="URL to Koeppen-Geiger-ASCII.txt")
    ap.add_argument("--kg-path", default=str(RESULTS_DIR / "Koeppen-Geiger-ASCII.txt"),
                    help="Local path for KG ASCII file (downloaded if missing).")
    args = ap.parse_args()
    parquet_path = PROCESSED_DIR / "aerocalib_hourly.parquet"
    kg_path = type(RESULTS_DIR / "x")(args.kg_path)  # Path-like via same class as RESULTS_DIR

    # 1) Airport coords
    coords = extract_airport_coords_from_parquet(parquet_path)
    print(f"Found coordinates for {len(coords)} airports.")

    # 2) Download & load KG ASCII
    download_if_missing(args.kg_url, kg_path)
    kg_df = load_koppen_ascii(kg_path)
    print(f"Loaded KG grid points: {len(kg_df):,}")

    # 3) Build airport -> regime map
    airport_to_regime = {}
    for regime, airports in CLIMATE_ZONES.items():
        for a in airports:
            airport_to_regime[a] = regime

    coords["regime"] = coords["airport"].map(airport_to_regime)
    coords = coords.dropna(subset=["regime"]).copy()

    # 4) Assign KG class to each airport
    kg_classes = []
    for row in coords.itertuples(index=False):
        kg = nearest_koppen_class(kg_df, row.lat, row.lon)
        kg_classes.append(kg)
    coords["koppen"] = kg_classes

    # 5) Aggregate counts per regime
    regimes_ordered = [
        "Northern Europe", "Mediterranean", "Continental",
        "East Asian", "Tropical", "Southern Hemisphere"
    ]

    regime_counts = {}
    for r in regimes_ordered:
        sub = coords[coords["regime"] == r]
        regime_counts[r] = Counter(sub["koppen"].tolist())

    # 6) LaTeX table
    def short_name(name: str) -> str:
        name = name.replace("Southern Hemisphere", r"S.\ Hemisphere")
        name = name.replace("Northern Europe", r"N.\ Europe")
        return name

    latex_lines = []
    latex_lines.append(r"\begin{table}[!t]")
    latex_lines.append(
        r"\caption{Approximate crosswalk between operational regimes and K\"{o}ppen--Geiger classes "
        r"(counts of airports per class). The K\"{o}ppen labels are reported for transparency; "
        r"our regimes are defined for operational similarity rather than climatological taxonomy.}"
    )
    latex_lines.append(r"\label{tab:koppen_crosswalk}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{lp{6.6cm}}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Operational regime & Dominant K\"{o}ppen--Geiger classes (count) \\")
    latex_lines.append(r"\midrule")

    for r in regimes_ordered:
        label = short_name(r)
        if r == "Tropical":
            label = r"Tropical / Convective-dominant"
        latex_lines.append(f"{label} & {format_counts(regime_counts[r])} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines)

    print("\n" + "=" * 70)
    print("LATEX TABLE tab:koppen_crosswalk (copy-paste):")
    print(latex_str)

    # 7) Save JSON (traceable: per-airport KG class too)
    out = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "16_koppen_crosswalk.py",
            "kg_url": args.kg_url,
            "kg_path": str(kg_path),
            "matching": "nearest grid point in (lat, lon) on 0.5-degree KG ASCII grid",
        },
        "airport_assignments": coords[["airport", "regime", "lat", "lon", "koppen"]].to_dict(orient="records"),
        "regime_counts": {r: dict(regime_counts[r]) for r in regimes_ordered},
        "latex": latex_str,
    }

    out_path = RESULTS_DIR / "koppen_crosswalk.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {out_path}")

    del kg_df, coords
    gc.collect()

if __name__ == "__main__":
    main()