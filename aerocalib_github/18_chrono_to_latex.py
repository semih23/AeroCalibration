#!/usr/bin/env python3
"""
Convert chronological_adaptation.json to LaTeX table rows.
Run after 17_chronological_adaptation.py completes.

Usage:
  python 18_chrono_to_latex.py
"""

import json
import sys
from pathlib import Path
from config import RESULTS_DIR

DISPLAY_WEEKS = [2, 8, 24]  # weeks to show in the table (3 per task)

PAIR_ORDER = [
    ("Southern Hemisphere", "Tropical"),
    ("Northern Europe", "Mediterranean"),
]
TASK_ORDER = ["T1", "T4"]
PAIR_LABELS = {
    ("Southern Hemisphere", "Tropical"): r"\textit{S.Hem $\rightarrow$ Tropical (hard shift)}",
    ("Northern Europe", "Mediterranean"): r"\textit{N.Eur $\rightarrow$ Mediterranean (mild shift)}",
}

def fmt_ece(v):
    if v is None or v != v:
        return "---"
    return f".{v:.4f}"[1:]  # e.g. .0069

def fmt_auc(v):
    if v is None or v != v:
        return "---"
    return f".{v:.3f}"[1:]  # e.g. .942

def fmt_pct(v):
    if v is None or v != v:
        return "---"
    return f"{v * 100:.0f}\\%"

def fmt_n(v):
    if v is None:
        return "---"
    return f"{v:,}"

def main():
    json_path = RESULTS_DIR / "chronological_adaptation.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run 17_chronological_adaptation.py first.")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    # Index experiments
    exp_map = {}
    for exp in data["experiments"]:
        key = (exp["source_zone"], exp["target_zone"], exp["task"])
        exp_map[key] = exp

    # Print table body
    print("% === AUTO-GENERATED LaTeX rows ===")
    print("% Paste between \\midrule lines in the table")
    print()

    for src, tgt in PAIR_ORDER:
        label = PAIR_LABELS.get((src, tgt), f"{src} -> {tgt}")
        print(f"\\multicolumn{{7}}{{l}}{{{label}}} \\\\")

        for tid in TASK_ORDER:
            key = (src, tgt, tid)
            exp = exp_map.get(key)
            if exp is None:
                print(f"% WARNING: no data for {key}")
                continue

            chrono = exp["chronological"]
            # find the weeks we want
            week_map = {w["weeks"]: w for w in chrono}

            rows = []
            for wk in DISPLAY_WEEKS:
                # find closest available week
                best = min(week_map.keys(), key=lambda x: abs(x - wk))
                rows.append(week_map[best])

            first = True
            for w in rows:
                prefix = f"\\multirow{{{len(rows)}}}{{*}}{{{tid}}}" if first else ""
                first = False

                r = w.get("recal_only") or {}
                ft = w.get("ft_recal") or {}

                line = (
                    f" {prefix}\n"
                    f" & {w['weeks']} "
                    f" & {fmt_ece(r.get('ece'))} & {fmt_pct(r.get('R_ece'))}"
                    f" & {fmt_auc(ft.get('auc'))} & {fmt_pct(ft.get('R_auc'))}"
                    f" & {fmt_n(w['n_samples'])} \\\\"
                )
                print(line)

        if (src, tgt) != PAIR_ORDER[-1]:
            print("\\midrule")

    # Also print prose numbers
    print("\n\n% === KEY NUMBERS FOR PROSE ===")
    all_recal_2wk = []
    all_recal_8wk = []
    all_ft_8wk_auc_r = []

    for src, tgt in PAIR_ORDER:
        for tid in TASK_ORDER:
            key = (src, tgt, tid)
            exp = exp_map.get(key)
            if not exp:
                continue
            week_map = {w["weeks"]: w for w in exp["chronological"]}
            for wk_target, coll in [(2, all_recal_2wk), (8, all_recal_8wk)]:
                best = min(week_map.keys(), key=lambda x: abs(x - wk_target))
                w = week_map[best]
                r = w.get("recal_only") or {}
                if r.get("R_ece") is not None:
                    coll.append(r["R_ece"])
            best8 = min(week_map.keys(), key=lambda x: abs(x - 8))
            ft8 = (week_map[best8].get("ft_recal") or {})
            if ft8.get("R_auc") is not None:
                all_ft_8wk_auc_r.append(ft8["R_auc"])

    if all_recal_2wk:
        print(f"% Recal R_ECE at ~2 weeks: {min(all_recal_2wk)*100:.0f}--{max(all_recal_2wk)*100:.0f}%")
    if all_recal_8wk:
        print(f"% Recal R_ECE at ~8 weeks: {min(all_recal_8wk)*100:.0f}--{max(all_recal_8wk)*100:.0f}%")
    if all_ft_8wk_auc_r:
        print(f"% FT+R R_AUC at ~8 weeks:  {min(all_ft_8wk_auc_r)*100:.0f}--{max(all_ft_8wk_auc_r)*100:.0f}%")

if __name__ == "__main__":
    main()
