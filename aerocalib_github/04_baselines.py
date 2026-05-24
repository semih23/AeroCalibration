#!/usr/bin/env python3
"""Computes various baselines for comparison with XGBoost."""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (
    PROCESSED_DIR, RESULTS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight, 
    get_feature_columns, get_lag_only_features,
    clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

def load_data():
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'
    df = pd.read_parquet(parquet_path)
    df['year'] = df['datetime'].dt.year.astype('int16')
    
    train_mask = (df['year'] >= TRAIN_YEARS[0]) & (df['year'] <= TRAIN_YEARS[1])
    val_mask = (df['year'] >= VAL_YEARS[0]) & (df['year'] <= VAL_YEARS[1])
    test_mask = (df['year'] >= TEST_YEARS[0]) & (df['year'] <= TEST_YEARS[1])
    
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

def compute_baselines(train_df, val_df, test_df, task_id):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'
    current_col = f'{task_id}_current'
    
    feature_cols = get_feature_columns()
    available_features = [f for f in feature_cols if f in train_df.columns]

    # Prepare data
    train_data = train_df.dropna(subset=[label_col])
    test_data = test_df.dropna(subset=[label_col])
    
    X_train = train_data[available_features].values.astype(np.float32)
    y_train = train_data[label_col].values.astype(np.float32)
    X_test = test_data[available_features].values.astype(np.float32)
    y_test = test_data[label_col].values.astype(np.float32)
    
    results = {}
    prevalence = y_train.mean()
    
    # Climatology
    results['Climatology'] = {'roc_auc': 0.500, 'pr_auc': float(prevalence)}
    
    # Hourly Climatology
    train_hourly_prev = train_data.groupby(train_data['datetime'].dt.hour)[label_col].mean()
    test_hours = test_data['datetime'].dt.hour
    y_prob_hourly = test_hours.map(train_hourly_prev).fillna(prevalence).values.astype(np.float32)
    
    try:
        results['Hourly Climatology'] = {
            'roc_auc': float(roc_auc_score(y_test, y_prob_hourly)),
            'pr_auc': float(average_precision_score(y_test, y_prob_hourly))
        }
    except:
        results['Hourly Climatology'] = {'roc_auc': 0.5, 'pr_auc': float(prevalence)}
    
    del y_prob_hourly
    gc.collect()
    
    # Persistence (1h)
    if current_col in test_data.columns:
        y_persist = test_data[current_col].values.astype(np.float32)
        mask_valid = ~np.isnan(y_persist) & ~np.isnan(y_test)
        
        if mask_valid.sum() > 100 and y_test[mask_valid].sum() > 0:
            try:
                results['Persistence (1h)'] = {
                    'roc_auc': float(roc_auc_score(y_test[mask_valid], y_persist[mask_valid])),
                    'pr_auc': float(average_precision_score(y_test[mask_valid], y_persist[mask_valid]))
                }
            except:
                results['Persistence (1h)'] = {'roc_auc': 0.5, 'pr_auc': float(prevalence)}
    
    # Persistence (h=horizon)
    if horizon > 1 and current_col in test_data.columns:
        y_persist_h = test_data.groupby('airport')[current_col].shift(horizon - 1)
        mask_valid = y_persist_h.notna() & test_data[label_col].notna()
        
        if mask_valid.sum() > 100:
            try:
                y_t = test_data.loc[mask_valid, label_col].values.astype(np.float32)
                y_p = y_persist_h[mask_valid].values.astype(np.float32)
                if y_t.sum() > 0:
                    results[f'Persistence (h={horizon})'] = {
                        'roc_auc': float(roc_auc_score(y_t, y_p)),
                        'pr_auc': float(average_precision_score(y_t, y_p))
                    }
            except:
                pass
    
    # Logistic Regression - with imputation
    print(f"  Training Logistic Regression...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_imputed).astype(np.float32)
    
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, 
                           random_state=RANDOM_SEED, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
    
    results['Logistic Reg. (L2)'] = {
        'roc_auc': float(roc_auc_score(y_test, y_prob_lr)),
        'pr_auc': float(average_precision_score(y_test, y_prob_lr))
    }
    
    del lr, y_prob_lr
    gc.collect()
    
    # Random Forest - with imputed data
    print(f"  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced',
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train_imputed, y_train)
    y_prob_rf = rf.predict_proba(X_test_imputed)[:, 1].astype(np.float32)
    
    results['Random Forest'] = {
        'roc_auc': float(roc_auc_score(y_test, y_prob_rf)),
        'pr_auc': float(average_precision_score(y_test, y_prob_rf))
    }
    
    del rf, y_prob_rf, X_train_imputed, X_test_imputed, X_train_scaled, X_test_scaled
    gc.collect()
    
    # Lag-only XGBoost
    print(f"  Training Lag-only XGBoost...")
    lag_features = get_lag_only_features(task_id)
    available_lag = [f for f in lag_features if f in train_df.columns]
    
    train_lag = train_data.dropna(subset=available_lag)
    test_lag = test_data.dropna(subset=available_lag)
    
    X_train_lag = train_lag[available_lag].values.astype(np.float32)
    y_train_lag = train_lag[label_col].values.astype(np.float32)
    X_test_lag = test_lag[available_lag].values.astype(np.float32)
    y_test_lag = test_lag[label_col].values.astype(np.float32)
    
    scale_pos_weight = compute_scale_pos_weight(y_train_lag.mean())
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    params.pop('early_stopping_rounds', None)
    params.pop('random_state', None)
    
    lag_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    lag_model.fit(X_train_lag, y_train_lag, verbose=False)
    y_prob_lag = lag_model.predict_proba(X_test_lag)[:, 1].astype(np.float32)
    
    results['Lag-only model'] = {
        'roc_auc': float(roc_auc_score(y_test_lag, y_prob_lag)),
        'pr_auc': float(average_precision_score(y_test_lag, y_prob_lag)),
        'n_features': len(available_lag)
    }
    
    del lag_model, y_prob_lag, X_train_lag, X_test_lag
    gc.collect()
    
    # Load main XGBoost results
    main_results_path = RESULTS_DIR / 'main_results.json'
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_results = json.load(f)
        if task_id in main_results.get('tasks', {}):
            xgb_metrics = main_results['tasks'][task_id]['test_metrics']
            results['XGBoost'] = {
                'roc_auc': xgb_metrics['roc_auc'],
                'pr_auc': xgb_metrics['pr_auc']
            }
    
    # Compute delta vs persistence
    if 'Persistence (1h)' in results and 'XGBoost' in results:
        delta = results['XGBoost']['roc_auc'] - results['Persistence (1h)']['roc_auc']
        results['delta_vs_persistence_pp'] = float(delta * 100)
    
    return results

def compute_signal_beyond_persistence(train_df, val_df, test_df, task_id):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'
    
    full_features = get_feature_columns()
    lag_features = get_lag_only_features(task_id)
    
    available_full = [f for f in full_features if f in train_df.columns]
    available_lag = [f for f in lag_features if f in train_df.columns]

    # Prepare paired data
    train_paired = train_df.dropna(subset=[label_col])
    test_paired = test_df.dropna(subset=[label_col])
    
    X_train_full = train_paired[available_full].values.astype(np.float32)
    X_train_lag = train_paired[available_lag].values.astype(np.float32)
    y_train = train_paired[label_col].values.astype(np.float32)
    
    X_test_full = test_paired[available_full].values.astype(np.float32)
    X_test_lag = test_paired[available_lag].values.astype(np.float32)
    y_test = test_paired[label_col].values.astype(np.float32)
    
    scale_pos_weight = compute_scale_pos_weight(y_train.mean())
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    params.pop('early_stopping_rounds', None)
    params.pop('random_state', None)
    
    # Lag-only model
    print(f"  Training lag-only model on paired subset...")
    lag_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    lag_model.fit(X_train_lag, y_train, verbose=False)
    y_prob_lag = lag_model.predict_proba(X_test_lag)[:, 1]
    
    lag_auc = roc_auc_score(y_test, y_prob_lag)
    lag_pr = average_precision_score(y_test, y_prob_lag)
    
    del lag_model, y_prob_lag
    gc.collect()
    
    # Full model
    print(f"  Training full model on paired subset...")
    full_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    full_model.fit(X_train_full, y_train, verbose=False)
    y_prob_full = full_model.predict_proba(X_test_full)[:, 1]
    
    full_auc = roc_auc_score(y_test, y_prob_full)
    full_pr = average_precision_score(y_test, y_prob_full)
    
    del full_model, y_prob_full
    gc.collect()
    
    return {
        'task': task_id,
        'horizon': horizon,
        'n_paired_samples': int(len(y_test)),
        'lag_only': {'roc_auc': float(lag_auc), 'pr_auc': float(lag_pr), 'n_features': len(available_lag)},
        'full_features': {'roc_auc': float(full_auc), 'pr_auc': float(full_pr), 'n_features': len(available_full)},
        'delta_auc': float(full_auc - lag_auc),
        'delta_pr': float(full_pr - lag_pr)
    }

def main():
    print("\nLoading data...")
    print_memory("start")
    train_df, val_df, test_df = load_data()
    print_memory("after load")
    
    all_baselines = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '04_baselines.py',
            'latex_table': 'Table 2 (Extended Baseline Comparison)'
        },
        'tasks': {}
    }
    
    signal_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '04_baselines.py',
            'latex_table': 'Table (Signal Beyond Persistence)'
        },
        'tasks': {}
    }
    
    for task_id in TASKS.keys():
        print(f"\n{'='*60}")
        print(f"Computing baselines for {task_id}")
        print(f"{'='*60}")
        print_memory(f"before {task_id}")
        
        baselines = compute_baselines(train_df, val_df, test_df, task_id)
        all_baselines['tasks'][task_id] = baselines
        
        print(f"\n  Computing signal beyond persistence...")
        signal = compute_signal_beyond_persistence(train_df, val_df, test_df, task_id)
        signal_results['tasks'][task_id] = signal
        
        gc.collect()
        print_memory(f"after {task_id}")
    
    # Save results
    output_baselines = RESULTS_DIR / 'baselines.json'
    with open(output_baselines, 'w') as f:
        json.dump(all_baselines, f, indent=2)
    print(f"\nSaved: {output_baselines}")
    
    output_signal = RESULTS_DIR / 'signal_beyond_persistence.json'
    with open(output_signal, 'w') as f:
        json.dump(signal_results, f, indent=2)
    print(f"Saved: {output_signal}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR LATEX TABLE 2")
    for task_id, baselines in all_baselines['tasks'].items():
        horizon = TASKS[task_id].horizon
        print(f"\n{task_id} (h={horizon}):")
        print(f"  {'Method':<25} {'ROC-AUC':>10}")
        print(f"  {'-'*37}")
        for method, metrics in baselines.items():
            if isinstance(metrics, dict) and 'roc_auc' in metrics:
                print(f"  {method:<25} {metrics['roc_auc']:>10.3f}")
    
    print("\nBASELINE COMPARISON COMPLETE")
    
    # Final cleanup
    del train_df, val_df, test_df
    gc.collect()

if __name__ == "__main__":
    main()
