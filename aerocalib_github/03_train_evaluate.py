#!/usr/bin/env python3
"""Trains XGBoost models for all tasks and evaluates on test set."""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss
)

from config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, MAX_SCALE_POS_WEIGHT,
    BOOTSTRAP_BLOCK_SIZE, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CONFIDENCE,
    ECE_N_BINS, RANDOM_SEED, compute_scale_pos_weight, get_feature_columns,
    clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

def compute_ece(y_true, y_prob, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.percentile(y_prob, quantiles * 100)
    bins = np.unique(bins)
    
    bin_indices = np.digitize(y_prob, bins[1:-1])
    
    ece = 0.0
    n = len(y_true)
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += mask.sum() * np.abs(avg_accuracy - avg_confidence)
    
    return ece / n

def compute_all_metrics(y_true, y_prob):
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    
    if len(y_true) == 0 or y_true.sum() == 0:
        return None
    
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    nll = log_loss(y_true, y_prob_clipped)
    
    ece = compute_ece(y_true, y_prob)
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier': float(brier),
        'nll': float(nll),
        'ece': float(ece),
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'prevalence': float(y_true.mean())
    }

def block_bootstrap_ci_fast(y_true, y_prob, block_ids, n_resamples=300, confidence=0.95):
    """
    FAST block bootstrap using numpy indexing only.
    No DataFrame operations - pure numpy for speed and memory efficiency.
    
    block_ids: integer array where same block_id = same station-day
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    block_ids = np.asarray(block_ids)
    
    # Get unique blocks and create mapping
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)
    
    # Pre-compute indices for each block (memory efficient)
    block_to_indices = {}
    for block in unique_blocks:
        block_to_indices[block] = np.where(block_ids == block)[0]
    
    bootstrap_aucs = []
    bootstrap_prs = []
    bootstrap_briers = []
    
    rng = np.random.default_rng(RANDOM_SEED)
    
    for i in range(n_resamples):
        # Sample blocks with replacement
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        
        # Collect indices
        sampled_indices = []
        for block in sampled_blocks:
            sampled_indices.extend(block_to_indices[block])
        
        sampled_indices = np.array(sampled_indices)
        
        y_t = y_true[sampled_indices]
        y_p = y_prob[sampled_indices]
        
        # Skip if degenerate
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        
        try:
            bootstrap_aucs.append(roc_auc_score(y_t, y_p))
            bootstrap_prs.append(average_precision_score(y_t, y_p))
            bootstrap_briers.append(brier_score_loss(y_t, y_p))
        except:
            continue
    
    alpha = 1 - confidence
    ci_results = {}
    
    for metric, values in [('roc_auc', bootstrap_aucs), 
                           ('pr_auc', bootstrap_prs), 
                           ('brier', bootstrap_briers)]:
        if len(values) > 10:
            values = np.array(values)
            ci_results[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, alpha/2 * 100)),
                'ci_upper': float(np.percentile(values, (1 - alpha/2) * 100)),
                'n_valid': len(values)
            }
    
    return ci_results

def load_data_for_task(task_id):
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Data file not found: {parquet_path}")
    
    feature_cols = get_feature_columns()
    task_config = TASKS[task_id]
    label_col = f'{task_id}_h{task_config.horizon}'
    
    # Columns to load
    needed_cols = feature_cols + [label_col, 'datetime', 'airport', 'year']
    
    # Load parquet with only needed columns
    df = pd.read_parquet(parquet_path, columns=needed_cols)
    
    # Add year if not present
    if 'year' not in df.columns:
        df['year'] = df['datetime'].dt.year
    
    # Split by years
    train_mask = (df['year'] >= TRAIN_YEARS[0]) & (df['year'] <= TRAIN_YEARS[1])
    val_mask = (df['year'] >= VAL_YEARS[0]) & (df['year'] <= VAL_YEARS[1])
    test_mask = (df['year'] >= TEST_YEARS[0]) & (df['year'] <= TEST_YEARS[1])
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    del df
    gc.collect()
    
    return train_df, val_df, test_df

def train_task(task_id):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'
    feature_cols = get_feature_columns()
    
    print(f"\n{'='*60}")
    print(f"Training {task_id}: {task_config.name} (h={horizon})")
    print(f"{'='*60}")
    
    print_memory("start")
    
    # Load data for this task only
    print("Loading data...")
    train_df, val_df, test_df = load_data_for_task(task_id)
    
    available_features = [f for f in feature_cols if f in train_df.columns]
    print(f"Features: {len(available_features)}")

    # Prepare data - drop rows with missing labels only
    train_data = train_df.dropna(subset=[label_col])
    val_data = val_df.dropna(subset=[label_col])
    test_data = test_df.dropna(subset=[label_col])
    
    # Convert to numpy arrays immediately (more memory efficient for XGBoost)
    X_train = train_data[available_features].values.astype(np.float32)
    y_train = train_data[label_col].values.astype(np.float32)
    
    X_val = val_data[available_features].values.astype(np.float32)
    y_val = val_data[label_col].values.astype(np.float32)
    
    X_test = test_data[available_features].values.astype(np.float32)
    y_test = test_data[label_col].values.astype(np.float32)
    
    # For bootstrap: create block IDs as integers
    test_data = test_data.reset_index(drop=True)
    test_dates = pd.to_datetime(test_data['datetime']).dt.date
    test_airports = test_data['airport'].astype(str)
    block_strings = test_airports + '_' + test_dates.astype(str)
    block_ids = pd.factorize(block_strings)[0]  # Integer block IDs
    
    # Clear DataFrames
    del train_df, val_df, test_df, train_data, val_data
    del test_dates, test_airports, block_strings
    gc.collect()
    
    print_memory("after data prep")
    
    prevalence = y_train.mean()
    scale_pos_weight = compute_scale_pos_weight(prevalence)
    
    print(f"Train: {len(X_train):,} samples, {int(y_train.sum()):,} positive ({prevalence*100:.3f}%)")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # XGBoost parameters
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    early_stopping = params.pop('early_stopping_rounds')
    random_state = params.pop('random_state')
    
    start_time = time.time()
    
    model = xgb.XGBClassifier(
        **params,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.1f}s")
    
    # Save model
    model_path = MODELS_DIR / f'xgb_{task_id}_h{horizon}.json'
    model.save_model(str(model_path))
    print(f"Model saved: {model_path}")
    
    # Predictions
    y_prob_test = model.predict_proba(X_test)[:, 1].astype(np.float32)
    
    # Clear training data
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    print_memory("after training")
    
    # Metrics
    test_metrics = compute_all_metrics(y_test, y_prob_test)
    
    print(f"\nTest Metrics (Raw):")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"  Brier: {test_metrics['brier']:.6f}")
    print(f"  ECE: {test_metrics['ece']:.6f}")
    
    # Bootstrap CI
    print(f"\nComputing block bootstrap CI (n={BOOTSTRAP_N_RESAMPLES})...")
    ci_results = block_bootstrap_ci_fast(
        y_test, y_prob_test, block_ids,
        n_resamples=BOOTSTRAP_N_RESAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE
    )
    
    if 'roc_auc' in ci_results:
        ci = ci_results['roc_auc']
        print(f"  ROC-AUC 95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    # Clear test data
    del X_test, y_test, y_prob_test, block_ids, test_data, model
    gc.collect()
    
    print_memory("end")
    
    return {
        'task': task_id,
        'horizon': horizon,
        'name': task_config.name,
        'training_time_seconds': float(training_time),
        'n_features': len(available_features),
        'train_samples': int(len(y_train) if 'y_train' in dir() else 0),
        'train_positive': 0,  # Already cleared
        'train_prevalence': float(prevalence),
        'test_metrics': test_metrics,
        'bootstrap_ci': ci_results,
        'model_path': str(model_path),
    }

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '03_train_evaluate.py',
            'latex_table': 'Table 3 (Test Set Performance)',
            'xgboost_params': XGBOOST_PARAMS,
            'bootstrap_config': {
                'block_size': BOOTSTRAP_BLOCK_SIZE,
                'n_resamples': BOOTSTRAP_N_RESAMPLES,
                'confidence': BOOTSTRAP_CONFIDENCE
            }
        },
        'tasks': {}
    }
    
    for task_id in TASKS.keys():
        try:
            result = train_task(task_id)
            all_results['tasks'][task_id] = result
            
            # Force cleanup between tasks
            gc.collect()
            
        except Exception as e:
            print(f"\nERROR training {task_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_json = RESULTS_DIR / 'main_results.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_json}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR LATEX TABLE 3 (Test Set Performance)")
    print(f"{'Task':<12} {'Horizon':>8} {'ROC-AUC':>10} {'95% CI':>20} {'PR-AUC':>10} {'Prev':>8}")
    for task_id, result in all_results['tasks'].items():
        metrics = result['test_metrics']
        ci = result['bootstrap_ci'].get('roc_auc', {})
        ci_str = f"[{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}]"
        prev_str = f"{metrics['prevalence']*100:.2f}%"
        
        print(f"{task_id:<12} h={result['horizon']:<5} {metrics['roc_auc']:>10.3f} {ci_str:>20} {metrics['pr_auc']:>10.3f} {prev_str:>8}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
if __name__ == "__main__":
    main()
