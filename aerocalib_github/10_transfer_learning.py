#!/usr/bin/env python3
"""Evaluates transfer learning from one climate zone to another."""

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
    PROCESSED_DIR, RESULTS_DIR, CLIMATE_ZONES,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight,
    get_feature_columns, clear_memory, print_memory
)

def load_data_efficient():
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'

    # Determine needed columns
    feature_cols = get_feature_columns()
    needed_cols = feature_cols + ['datetime', 'airport', 'T1_h6']  # T1 is used for transfer

    # Load only needed columns
    df = pd.read_parquet(parquet_path, columns=needed_cols)
    df['year'] = df['datetime'].dt.year.astype('int16')

    # Optimize dtypes
    for col in feature_cols:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

    print_memory("After loading data")
    return df

def transfer_experiment(df, task_id, feature_cols, source_zone, target_zone,
                        target_fractions, n_seeds=10):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'

    print(f"\n{'=' * 60}")
    print(f"Transfer: {source_zone} -> {target_zone} ({task_id})")
    print(f"{'=' * 60}")

    source_airports = CLIMATE_ZONES.get(source_zone, [])
    target_airports = CLIMATE_ZONES.get(target_zone, [])

    if not source_airports or not target_airports:
        print(f"  ERROR: Invalid zones")
        return None

    available_features = [f for f in feature_cols if f in df.columns]
    print(f"  Using {len(available_features)} features")

    # Source training data - use efficient masking
    source_mask = (
            (df['year'] >= TRAIN_YEARS[0]) &
            (df['year'] <= TRAIN_YEARS[1]) &
            (df['airport'].isin(source_airports)) &
            (df[label_col].notna())
    )
    source_train = df.loc[source_mask, available_features + [label_col]].copy()
    print(f"  Source train samples: {len(source_train):,}")

    if len(source_train) < 1000:
        print(f"  ERROR: Not enough source data")
        del source_train
        clear_memory()
        return None

    # Convert to numpy arrays immediately (memory efficient)
    X_source = source_train[available_features].values.astype(np.float32)
    y_source = source_train[label_col].values.astype(np.float32)
    del source_train
    clear_memory()

    if y_source.sum() == 0:
        print(f"  ERROR: No positive samples in source")
        return None

    print(f"  Source positive rate: {y_source.mean():.4f}")

    # Target data
    target_mask = df['airport'].isin(target_airports) & df[label_col].notna()
    target_all = df.loc[target_mask].copy()
    target_all = target_all.sort_values('datetime')

    print(f"  Target total samples: {len(target_all):,}")

    # Model parameters
    scale_pos_weight = compute_scale_pos_weight(y_source.mean())
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    params.pop('early_stopping_rounds', None)
    params.pop('random_state', None)

    # Train source model
    print(f"\n  Training source model...")
    print_memory("Before source model")

    source_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    source_model.fit(X_source, y_source, verbose=False)

    # Free source training data
    del X_source, y_source
    clear_memory()
    print_memory("After source model training")

    # Target test set (2023-2024)
    test_mask = (target_all['year'] >= TEST_YEARS[0]) & (target_all['year'] <= TEST_YEARS[1])
    target_test = target_all.loc[test_mask]

    X_target_test = target_test[available_features].values.astype(np.float32)
    y_target_test = target_test[label_col].values.astype(np.float32)

    if len(y_target_test) < 100 or y_target_test.sum() == 0:
        print(f"  ERROR: Not enough test samples or no positives")
        del source_model, target_all, target_test
        clear_memory()
        return None

    # Source-only performance on target test
    y_prob_source = source_model.predict_proba(X_target_test)[:, 1]
    source_only_auc = roc_auc_score(y_target_test, y_prob_source)
    print(f"  Source-only AUC on target: {source_only_auc:.3f}")
    del y_prob_source

    # Target training pool (2010-2022)
    train_pool_mask = (target_all['year'] >= TRAIN_YEARS[0]) & (target_all['year'] <= VAL_YEARS[1])
    target_train_pool = target_all.loc[train_pool_mask]

    X_target_pool = target_train_pool[available_features].values.astype(np.float32)
    y_target_pool = target_train_pool[label_col].values.astype(np.float32)

    # Free target_all DataFrame
    del target_all, target_train_pool
    clear_memory()

    if len(y_target_pool) < 100:
        print(f"  ERROR: Not enough target training pool")
        del source_model
        clear_memory()
        return None

    # Oracle model (100% target data)
    print(f"  Training oracle model (100% target data)...")
    oracle_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    oracle_model.fit(X_target_pool, y_target_pool, verbose=False)
    y_prob_oracle = oracle_model.predict_proba(X_target_test)[:, 1]
    oracle_auc = roc_auc_score(y_target_test, y_prob_oracle)
    print(f"  Oracle AUC: {oracle_auc:.3f}")

    del oracle_model, y_prob_oracle
    clear_memory()

    results = {
        'source_zone': source_zone,
        'target_zone': target_zone,
        'source_airports': source_airports,
        'target_airports': target_airports,
        'n_source_train': int(len(X_target_pool)),  # Will be updated
        'n_target_train_pool': int(len(X_target_pool)),
        'n_target_test': int(len(X_target_test)),
        'n_features': len(available_features),
        'source_positive_rate': float(y_target_pool.mean()),
        'target_positive_rate': float(y_target_test.mean()),
        'source_only_auc': float(source_only_auc),
        'oracle_auc': float(oracle_auc),
        'transfer_results': []
    }

    # Transfer experiments with different fractions
    for frac in target_fractions:
        if frac == 0:
            results['transfer_results'].append({
                'target_fraction': 0.0,
                'target_fraction_pct': "0%",
                'mean_auc': float(source_only_auc),
                'std_auc': 0.0,
                'gap_recovery_pct': 0.0
            })
            continue

        if frac == 1.0:
            results['transfer_results'].append({
                'target_fraction': 1.0,
                'target_fraction_pct': "100%",
                'mean_auc': float(oracle_auc),
                'std_auc': 0.0,
                'gap_recovery_pct': 100.0
            })
            continue

        print(f"\n  Fine-tuning with {frac * 100:.0f}% target data...")

        n_samples = max(50, int(len(X_target_pool) * frac))

        frac_aucs = []
        for seed in range(RANDOM_SEED, RANDOM_SEED + n_seeds):
            np.random.seed(seed)

            # Stratified sampling
            pos_idx = np.where(y_target_pool == 1)[0]
            neg_idx = np.where(y_target_pool == 0)[0]

            n_pos = max(1, int(len(pos_idx) * frac))
            n_neg = max(1, n_samples - n_pos)

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            sampled_pos = np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False)
            sampled_neg = np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)
            sampled_idx = np.concatenate([sampled_pos, sampled_neg])

            X_finetune = X_target_pool[sampled_idx]
            y_finetune = y_target_pool[sampled_idx]

            if y_finetune.sum() == 0:
                continue

            try:
                # Fine-tune from source model
                adapted_model = xgb.XGBClassifier(**params, random_state=seed, use_label_encoder=False)
                adapted_model.fit(
                    X_finetune, y_finetune,
                    xgb_model=source_model.get_booster(),
                    verbose=False
                )

                y_prob_adapted = adapted_model.predict_proba(X_target_test)[:, 1]
                frac_aucs.append(roc_auc_score(y_target_test, y_prob_adapted))

                del adapted_model
            except Exception as e:
                print(f"    Seed {seed} failed: {e}")
                continue

        # Cleanup after each fraction
        clear_memory()

        if not frac_aucs:
            print(f"    All seeds failed for {frac * 100:.0f}%")
            continue

        mean_auc = np.mean(frac_aucs)
        std_auc = np.std(frac_aucs)

        gap = oracle_auc - source_only_auc
        if gap > 0:
            recovery = (mean_auc - source_only_auc) / gap * 100
        else:
            recovery = 100.0 if mean_auc >= oracle_auc else 0.0

        print(f"    AUC: {mean_auc:.3f} ± {std_auc:.3f}, Recovery: {recovery:.0f}%")

        results['transfer_results'].append({
            'target_fraction': float(frac),
            'target_fraction_pct': f"{frac * 100:.0f}%",
            'mean_auc': float(mean_auc),
            'std_auc': float(std_auc),
            'gap_recovery_pct': float(recovery)
        })

    # Final cleanup
    del source_model, X_target_pool, y_target_pool, X_target_test, y_target_test
    clear_memory()
    print_memory("After experiment cleanup")

    return results

def main():
    print_memory("Start")

    print("\nLoading data...")
    df = load_data_efficient()
    print(f"Loaded {len(df):,} rows")

    feature_cols = get_feature_columns()
    print(f"Feature columns: {len(feature_cols)}")

    target_fractions = [0.0, 0.01, 0.05, 0.10, 0.20, 1.0]

    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '10_transfer_learning.py',
            'latex_table': 'Table 9 (Transfer Learning)',
            'target_fractions': target_fractions,
            'n_seeds': 10,
            'memory_optimized': True
        },
        'experiments': []
    }

    # Experiment 1: Southern Hemisphere -> Tropical
    result = transfer_experiment(
        df, 'T1', feature_cols,
        source_zone='Southern Hemisphere',
        target_zone='Tropical',
        target_fractions=target_fractions
    )

    if result:
        all_results['experiments'].append(result)
        print("\n✓ Experiment 1 completed successfully!")
    else:
        print("\n✗ Experiment 1 failed!")

    clear_memory()

    # Experiment 2: Northern Europe -> Mediterranean (backup)
    result2 = transfer_experiment(
        df, 'T1', feature_cols,
        source_zone='Northern Europe',
        target_zone='Mediterranean',
        target_fractions=target_fractions
    )

    if result2:
        all_results['experiments'].append(result2)
        print("\n✓ Experiment 2 completed successfully!")

    # Free main DataFrame
    del df
    clear_memory()

    # Save results
    output_json = RESULTS_DIR / 'transfer.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_json}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR LATEX TABLE 9")
    for exp in all_results['experiments']:
        print(f"\n{exp['source_zone']} -> {exp['target_zone']}")
        print(f"  Source train samples: {exp['n_source_train']:,}")
        print(f"  Target test samples: {exp['n_target_test']:,}")
        print(f"  Features used: {exp['n_features']}")
        print(f"\n  {'Target %':<15} {'AUC':>8} {'Std':>8} {'Recovery':>12}")
        print("  " + "-" * 46)

        for r in exp['transfer_results']:
            if r['target_fraction'] == 0:
                print(f"  {'0% (source)':<15} {r['mean_auc']:>8.3f} {'--':>8} {'0%':>12}")
            elif r['target_fraction'] == 1.0:
                print(f"  {'100% (oracle)':<15} {r['mean_auc']:>8.3f} {'--':>8} {'100%':>12}")
            else:
                print(
                    f"  {r['target_fraction_pct']:<15} {r['mean_auc']:>8.3f} {r['std_auc']:>8.3f} {r['gap_recovery_pct']:>11.0f}%")

    print_memory("End")
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING COMPLETE")
if __name__ == "__main__":
    main()