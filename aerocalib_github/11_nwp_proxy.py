#!/usr/bin/env python3
"""Compares METAR-only model with METAR+ERA5 augmented model."""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import roc_auc_score

from config import (
    PROCESSED_DIR, RESULTS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight,
    get_feature_columns, clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

def get_era5_features():
    era5_vars = [
        'era5_t2m', 'era5_d2m', 'era5_sp', 'era5_u10', 'era5_v10',
        'era5_tcc', 'era5_cape', 'era5_tcwv'
    ]

    era5_features = []
    for var in era5_vars:
        era5_features.append(var)
        for lag in [1, 3, 6]:
            era5_features.append(f'{var}_lag{lag}')

    return era5_features

def load_data_for_task(task_id, metar_features):
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'

    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'

    # Check for ERA5 features
    era5_features = get_era5_features()

    # Columns to load
    needed_cols = ['datetime'] + metar_features + [label_col]

    # Try to add ERA5 columns if they exist
    try:
        # Read just column names first
        sample = pd.read_parquet(parquet_path, columns=['datetime'])
        all_cols = pd.read_parquet(parquet_path).columns.tolist()
        available_era5 = [f for f in era5_features if f in all_cols]
        needed_cols = needed_cols + available_era5
        del sample
    except:
        available_era5 = []

    # Make unique
    needed_cols = list(set(needed_cols))

    # Load data
    df = pd.read_parquet(parquet_path, columns=needed_cols)
    df['year'] = df['datetime'].dt.year.astype('int16')

    # Optimize dtypes
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

    # Split by year
    train_mask = (df['year'] >= TRAIN_YEARS[0]) & (df['year'] <= TRAIN_YEARS[1])
    val_mask = (df['year'] >= VAL_YEARS[0]) & (df['year'] <= VAL_YEARS[1])
    test_mask = (df['year'] >= TEST_YEARS[0]) & (df['year'] <= TEST_YEARS[1])

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    del df
    clear_memory()

    return train_df, val_df, test_df, available_era5

def compare_metar_vs_era5(train_df, val_df, test_df, task_id, metar_features, era5_available):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'

    print(f"\n{'=' * 60}")
    print(f"NWP Proxy Comparison for {task_id}")
    print(f"{'=' * 60}")
    print_memory("Start of comparison")

    available_metar = [f for f in metar_features if f in train_df.columns]
    available_era5 = [f for f in era5_available if f in train_df.columns]

    print(f"  METAR features: {len(available_metar)}")
    print(f"  ERA5 features available: {len(available_era5)}")

    # Get data with valid labels only
    train_valid = train_df[train_df[label_col].notna()].copy()
    test_valid = test_df[test_df[label_col].notna()].copy()

    # Convert to numpy arrays (more memory efficient for XGBoost)
    X_train = train_valid[available_metar].values.astype(np.float32)
    y_train = train_valid[label_col].values.astype(np.float32)
    X_test = test_valid[available_metar].values.astype(np.float32)
    y_test = test_valid[label_col].values.astype(np.float32)

    # Free DataFrames (keep test_valid for combined model if needed)
    del train_valid
    clear_memory()

    # Model parameters
    scale_pos_weight = compute_scale_pos_weight(y_train.mean())
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    params.pop('early_stopping_rounds', None)
    params.pop('random_state', None)

    # Train METAR-only model
    print(f"\n  Training METAR-only model...")
    metar_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    metar_model.fit(X_train, y_train, verbose=False)

    y_prob_metar = metar_model.predict_proba(X_test)[:, 1]
    metar_auc = roc_auc_score(y_test, y_prob_metar)
    print(f"  METAR-only AUC: {metar_auc:.4f}")

    # Free METAR model
    del metar_model, X_train, y_prob_metar
    clear_memory()

    # Combined model if ERA5 available
    if len(available_era5) > 0:
        all_features = available_metar + available_era5

        # Reload for combined model (with ERA5)
        train_valid_c = train_df[train_df[label_col].notna()].copy()
        test_valid_c = test_df[test_df[label_col].notna()].copy()

        X_train_c = train_valid_c[all_features].values.astype(np.float32)
        y_train_c = train_valid_c[label_col].values.astype(np.float32)
        X_test_c = test_valid_c[all_features].values.astype(np.float32)
        y_test_c = test_valid_c[label_col].values.astype(np.float32)

        del train_valid_c, test_valid_c
        clear_memory()

        print(f"\n  Training METAR+ERA5 model...")
        combined_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
        combined_model.fit(X_train_c, y_train_c, verbose=False)

        y_prob_combined = combined_model.predict_proba(X_test_c)[:, 1]
        combined_auc = roc_auc_score(y_test_c, y_prob_combined)
        print(f"  METAR+ERA5 AUC: {combined_auc:.4f}")

        delta_auc = combined_auc - metar_auc
        print(f"  Delta AUC: {delta_auc:+.4f} ({delta_auc / metar_auc * 100:+.2f}%)")

        # Cleanup
        del combined_model, X_train_c, X_test_c, y_train_c, y_test_c, y_prob_combined
        clear_memory()

        return {
            'task': task_id,
            'horizon': horizon,
            'metar_only': {
                'n_features': len(available_metar),
                'n_test': int(len(y_test)),
                'roc_auc': float(metar_auc)
            },
            'metar_plus_era5': {
                'n_features': len(all_features),
                'n_era5_features': len(available_era5),
                'n_test': int(len(y_test)),
                'roc_auc': float(combined_auc)
            },
            'delta_auc': float(delta_auc),
            'delta_auc_pp': float(delta_auc * 100),
            'improvement_pct': float(delta_auc / metar_auc * 100)
        }
    else:
        print(f"  NOTE: No ERA5 features available. Reporting METAR-only results.")

        # Cleanup
        del y_train, X_test, y_test
        clear_memory()

        return {
            'task': task_id,
            'horizon': horizon,
            'metar_only': {
                'n_features': len(available_metar),
                'n_test': int(len(test_valid)),
                'roc_auc': float(metar_auc)
            },
            'metar_plus_era5': None,
            'delta_auc': 0.0,
            'delta_auc_pp': 0.0,
            'improvement_pct': 0.0,
            'note': 'ERA5 features not available in dataset'
        }

def main():
    print("\nNOTE: ERA5 is reanalysis, not operational NWP forecast.")
    print("This is a sensitivity analysis, not a true NWP comparison.")

    print_memory("Start")

    metar_features = get_feature_columns()

    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '11_nwp_proxy.py',
            'latex_table': 'Table 10 (NWP Proxy Comparison)',
            'caveat': 'ERA5 is reanalysis that assimilates METAR. Not a true NWP comparison.',
            'memory_optimized': True
        },
        'tasks': {}
    }

    for task_id in TASKS.keys():
        print(f"\n\nProcessing {task_id}...")
        print_memory(f"Before {task_id}")

        # Load data for this task only
        train_df, val_df, test_df, era5_available = load_data_for_task(task_id, metar_features)

        # Run comparison
        result = compare_metar_vs_era5(train_df, val_df, test_df, task_id, metar_features, era5_available)
        all_results['tasks'][task_id] = result

        # Cleanup
        del train_df, val_df, test_df
        clear_memory()
        print_memory(f"After {task_id}")

    # Save results
    output_json = RESULTS_DIR / 'nwp_proxy.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_json}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR LATEX TABLE 10")
    print(f"\n{'Task':>8} {'METAR AUC':>12} {'+ERA5 AUC':>12} {'ΔAUC (pp)':>12} {'Improv.':>10}")
    aucs = []
    deltas = []

    for task_id, result in all_results['tasks'].items():
        metar_auc = result['metar_only']['roc_auc']
        aucs.append(metar_auc)

        if result['metar_plus_era5']:
            era5_auc = result['metar_plus_era5']['roc_auc']
            delta = result['delta_auc_pp']
            improv = result['improvement_pct']
            deltas.append(result['delta_auc'])
            print(f"{task_id:>8} {metar_auc:>12.4f} {era5_auc:>12.4f} {delta:>+12.4f} {improv:>+9.2f}%")
        else:
            deltas.append(0.0)
            print(f"{task_id:>8} {metar_auc:>12.4f} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    mean_auc = np.mean(aucs)
    mean_delta = np.mean(deltas)
    print(f"{'Mean':>8} {mean_auc:>12.4f} {mean_auc + mean_delta:>12.4f} {mean_delta * 100:>+12.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("Small |ΔAUC| (<0.001) suggests METAR lag features already capture")
    print("most local signal relevant for threshold exceedance prediction.")
    print("This does NOT mean operational NWP is unnecessary for mesoscale context.")

    print_memory("End")
    print("\nNWP PROXY COMPARISON COMPLETE")

if __name__ == "__main__":
    main()