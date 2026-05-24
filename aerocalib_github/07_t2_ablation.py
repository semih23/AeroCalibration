#!/usr/bin/env python3
"""Validates that T2 prediction is not driven by label leakage."""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (
    PROCESSED_DIR, RESULTS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, LAG_HOURS, ROLLING_WINDOWS,
    compute_scale_pos_weight, get_feature_columns,
    clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

def get_no_fog_features():
    all_features = get_feature_columns()
    return [f for f in all_features if 'fog' not in f.lower()]

def get_no_fog_no_vis_features():
    features = get_no_fog_features()
    return [f for f in features if 'visibility' not in f.lower()]

def get_primitives_only_features():
    features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos', 'dayofyear']
    
    for var in ['temperature', 'dewpoint', 'pressure', 'wind_speed', 'wind_dir',
                'wind_u', 'wind_v', 'wind_dir_sin', 'wind_dir_cos']:
        for lag in LAG_HOURS:
            features.append(f'{var}_lag{lag}')
    
    for var in ['wind_u', 'wind_v', 'wind_speed']:
        for window in ROLLING_WINDOWS:
            features.append(f'{var}_roll{window}_mean')
            features.append(f'{var}_roll{window}_std')
    
    features.extend(['lat', 'lon', 'rwy_hdg'])
    return features

def load_data():
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'
    df = pd.read_parquet(parquet_path)
    df['year'] = df['datetime'].dt.year.astype('int16')
    
    train_mask = (df['year'] >= TRAIN_YEARS[0]) & (df['year'] <= TRAIN_YEARS[1])
    val_mask = (df['year'] >= VAL_YEARS[0]) & (df['year'] <= VAL_YEARS[1])
    test_mask = (df['year'] >= TEST_YEARS[0]) & (df['year'] <= TEST_YEARS[1])
    
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

def train_and_evaluate(train_df, val_df, test_df, feature_list, config_name):
    label_col = 'T2_h6'
    
    available_features = [f for f in feature_list if f in train_df.columns]

    train_data = train_df.dropna(subset=[label_col])
    test_data = test_df.dropna(subset=[label_col])
    
    X_train = train_data[available_features].values.astype(np.float32)
    y_train = train_data[label_col].values.astype(np.float32)
    X_test = test_data[available_features].values.astype(np.float32)
    y_test = test_data[label_col].values.astype(np.float32)
    
    prevalence = y_train.mean()
    scale_pos_weight = compute_scale_pos_weight(prevalence)
    
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    params.pop('early_stopping_rounds', None)
    params.pop('random_state', None)
    
    model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    model.fit(X_train, y_train, verbose=False)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    del model, X_train, y_train, X_test
    gc.collect()
    
    return {
        'config': config_name,
        'n_features': len(available_features),
        'n_test': int(len(y_test)),
        'n_positive': int(y_test.sum()),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }

def main():
    print("\nLoading data...")
    print_memory("start")
    train_df, val_df, test_df = load_data()
    print_memory("after load")
    
    ablation_configs = [
        ('Baseline (all features)', get_feature_columns()),
        ('No fog features', get_no_fog_features()),
        ('No fog + no visibility lags', get_no_fog_no_vis_features()),
        ('Primitives only (T, Td, p, wind) + time', get_primitives_only_features()),
    ]
    
    results = []
    
    for config_name, feature_list in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")
        
        result = train_and_evaluate(train_df, val_df, test_df, feature_list, config_name)
        results.append(result)
        
        print(f"  Features: {result['n_features']}")
        print(f"  ROC-AUC: {result['roc_auc']:.3f}")
        print(f"  PR-AUC: {result['pr_auc']:.3f}")
        
        gc.collect()
    
    baseline_auc = results[0]['roc_auc']
    for r in results:
        r['delta_auc'] = float(r['roc_auc'] - baseline_auc)
    
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '07_t2_ablation.py',
            'latex_table': 'Table 6 (T2 Leakage Checks)',
            'task': 'T2',
            'horizon': 6
        },
        'ablation_results': results
    }
    
    output_json = RESULTS_DIR / 't2_ablation.json'
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {output_json}")
    
    print("\n" + "=" * 70)
    print("T2 ABLATION STUDY COMPLETE")
    del train_df, val_df, test_df
    gc.collect()

if __name__ == "__main__":
    main()
