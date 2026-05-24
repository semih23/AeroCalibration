#!/usr/bin/env python3
"""Evaluates spatial generalization by holding out 9 airports."""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    HELDOUT_AIRPORTS, XGBOOST_PARAMS, RANDOM_SEED,
    compute_scale_pos_weight, get_feature_columns,
    clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

def load_data():
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'
    df = pd.read_parquet(parquet_path)
    df['year'] = df['datetime'].dt.year.astype('int16')
    return df

def evaluate_airport_heldout(df, task_id):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'
    feature_cols = get_feature_columns()

    print(f"\n{'='*60}")
    print(f"Airport-Held-Out Evaluation for {task_id}")
    print(f"{'='*60}")

    available_features = [f for f in feature_cols if f in df.columns]
    train_airports = [a for a in df['airport'].unique() if a not in HELDOUT_AIRPORTS]

    # Create masks
    train_mask = (
        (df['year'] >= TRAIN_YEARS[0]) &
        (df['year'] <= TRAIN_YEARS[1]) &
        (df['airport'].isin(train_airports))
    )
    val_mask = (
        (df['year'] >= VAL_YEARS[0]) &
        (df['year'] <= VAL_YEARS[1]) &
        (df['airport'].isin(train_airports))
    )
    test_mask = (
        (df['year'] >= TEST_YEARS[0]) &
        (df['year'] <= TEST_YEARS[1]) &
        (df['airport'].isin(HELDOUT_AIRPORTS))
    )

    # Prepare data
    train_df = df[train_mask].dropna(subset=[label_col])
    val_df = df[val_mask].dropna(subset=[label_col])
    test_df = df[test_mask].dropna(subset=[label_col])
    
    print(f"  Train airports: {len(train_airports)}")
    print(f"  Held-out airports: {len(HELDOUT_AIRPORTS)}")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples (held-out): {len(test_df):,}")

    if len(test_df) < 100:
        print("  ERROR: Not enough test samples")
        return None

    X_train = train_df[available_features].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.float32)
    X_val = val_df[available_features].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.float32)
    X_test = test_df[available_features].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)
    test_airports = test_df['airport'].values

    del train_df, val_df
    gc.collect()

    prevalence = y_train.mean()
    scale_pos_weight = compute_scale_pos_weight(prevalence)

    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    params.pop('early_stopping_rounds', None)
    params.pop('random_state', None)

    print(f"  Training XGBoost on {len(train_airports)} airports...")
    model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1].astype(np.float32)

    del model, X_train, y_train, X_val, y_val
    gc.collect()

    if y_test.sum() == 0:
        print("  WARNING: No positives in test set")
        return None

    micro_auc = roc_auc_score(y_test, y_prob)
    micro_pr = average_precision_score(y_test, y_prob)
    micro_brier = brier_score_loss(y_test, y_prob)

    print(f"\n  Micro metrics (pooled):")
    print(f"    ROC-AUC: {micro_auc:.3f}")
    print(f"    PR-AUC: {micro_pr:.3f}")

    # Per-airport results
    per_airport_results = []
    for airport in HELDOUT_AIRPORTS:
        mask = test_airports == airport
        if mask.sum() < 50:
            continue

        y_t = y_test[mask]
        y_p = y_prob[mask]

        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue

        try:
            per_airport_results.append({
                'airport': airport,
                'n_samples': int(mask.sum()),
                'n_positive': int(y_t.sum()),
                'roc_auc': float(roc_auc_score(y_t, y_p)),
                'pr_auc': float(average_precision_score(y_t, y_p)),
                'brier': float(brier_score_loss(y_t, y_p))
            })
        except:
            continue

    # Macro metrics
    if per_airport_results:
        aucs = [r['roc_auc'] for r in per_airport_results]
        prs = [r['pr_auc'] for r in per_airport_results]
        briers = [r['brier'] for r in per_airport_results]

        macro_auc = np.mean(aucs)
        macro_pr = np.mean(prs)
        macro_brier = np.mean(briers)
        median_auc = np.median(aucs)
        q1_auc = np.percentile(aucs, 25)
        q3_auc = np.percentile(aucs, 75)
    else:
        macro_auc = macro_pr = macro_brier = None
        median_auc = q1_auc = q3_auc = None

    # Load temporal split results
    main_results_path = RESULTS_DIR / 'main_results.json'
    temporal_auc = temporal_pr = None
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_results = json.load(f)
        if task_id in main_results.get('tasks', {}):
            temporal_auc = main_results['tasks'][task_id]['test_metrics']['roc_auc']
            temporal_pr = main_results['tasks'][task_id]['test_metrics']['pr_auc']

    del X_test, y_test, y_prob, test_df
    gc.collect()

    return {
        'task': task_id,
        'horizon': horizon,
        'n_train_airports': len(train_airports),
        'n_heldout_airports': len(HELDOUT_AIRPORTS),
        'heldout_airports': HELDOUT_AIRPORTS,
        'n_test_samples': int(len(test_airports)),
        'n_positive': int(sum(1 for r in per_airport_results for _ in [r['n_positive']])),
        'temporal_split': {'roc_auc': temporal_auc, 'pr_auc': temporal_pr},
        'micro': {'roc_auc': float(micro_auc), 'pr_auc': float(micro_pr), 'brier': float(micro_brier)},
        'macro': {
            'roc_auc': float(macro_auc) if macro_auc else None,
            'pr_auc': float(macro_pr) if macro_pr else None,
            'brier': float(macro_brier) if macro_brier else None,
            'n_airports_evaluated': len(per_airport_results)
        },
        'robust_summary': {
            'median_auc': float(median_auc) if median_auc else None,
            'iqr': [float(q1_auc), float(q3_auc)] if q1_auc else None
        },
        'per_airport': per_airport_results
    }

def main():
    print(f"\nHeld-out airports ({len(HELDOUT_AIRPORTS)}):")
    for a in HELDOUT_AIRPORTS:
        print(f"  - {a}")
    
    print("\nLoading data...")
    print_memory("start")
    df = load_data()
    print_memory("after load")
    
    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '06_airport_heldout.py',
            'latex_table': 'Table 5 (Airport-Held-Out Generalization)',
            'heldout_airports': HELDOUT_AIRPORTS
        },
        'tasks': {}
    }
    
    for task_id in TASKS.keys():
        result = evaluate_airport_heldout(df, task_id)
        if result:
            all_results['tasks'][task_id] = result
        gc.collect()
        print_memory(f"after {task_id}")
    
    output_json = RESULTS_DIR / 'airport_heldout.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_json}")
    
    print("\n" + "=" * 70)
    print("AIRPORT-HELD-OUT EVALUATION COMPLETE")
    del df
    gc.collect()

if __name__ == "__main__":
    main()
