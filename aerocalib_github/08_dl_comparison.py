#!/usr/bin/env python3
"""Compares XGBoost with deep learning architectures."""

import pandas as pd
import numpy as np
import json
import time
import sys
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from config import (
    PROCESSED_DIR, RESULTS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    XGBOOST_PARAMS, RANDOM_SEED, compute_scale_pos_weight, get_feature_columns,
    clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not available")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_mlp(X_train, y_train, X_val, y_val, X_test, epochs=50, batch_size=256, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    del X_train, X_val
    gc.collect()
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    del X_train_scaled, X_val_scaled
    gc.collect()
    
    input_dim = X_test_scaled.shape[1]
    model = MLP(input_dim).to(device)
    
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_t))
        
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t).squeeze()
            val_loss = criterion(val_outputs, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    training_time = time.time() - start_time
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_test_t)).cpu().numpy().squeeze()
    
    # Cleanup
    del X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return y_prob, training_time

def load_data():
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'
    df = pd.read_parquet(parquet_path)
    df['year'] = df['datetime'].dt.year.astype('int16')
    
    train_mask = (df['year'] >= TRAIN_YEARS[0]) & (df['year'] <= TRAIN_YEARS[1])
    val_mask = (df['year'] >= VAL_YEARS[0]) & (df['year'] <= VAL_YEARS[1])
    test_mask = (df['year'] >= TEST_YEARS[0]) & (df['year'] <= TEST_YEARS[1])
    
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

def compare_models(train_df, val_df, test_df, task_id, feature_cols, skip_slow=False):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'
    
    print(f"\n{'='*60}")
    print(f"Comparing models for {task_id}")
    print(f"{'='*60}")
    
    available_features = [f for f in feature_cols if f in train_df.columns]

    train_data = train_df.dropna(subset=[label_col])
    val_data = val_df.dropna(subset=[label_col])
    test_data = test_df.dropna(subset=[label_col])
    
    X_train = train_data[available_features].values.astype(np.float32)
    y_train = train_data[label_col].values.astype(np.float32)
    X_val = val_data[available_features].values.astype(np.float32)
    y_val = val_data[label_col].values.astype(np.float32)
    X_test = test_data[available_features].values.astype(np.float32)
    y_test = test_data[label_col].values.astype(np.float32)
    
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    results = {}
    
    # XGBoost
    print("  Training XGBoost...")
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = compute_scale_pos_weight(y_train.mean())
    params.pop('early_stopping_rounds', None)
    params.pop('random_state', None)
    
    start = time.time()
    xgb_model = xgb.XGBClassifier(**params, random_state=RANDOM_SEED, use_label_encoder=False)
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - start
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, y_prob_xgb)
    
    results['XGBoost'] = {'roc_auc': float(xgb_auc), 'time_seconds': float(xgb_time)}
    print(f"    AUC: {xgb_auc:.3f}, Time: {xgb_time:.1f}s")
    
    del xgb_model, y_prob_xgb
    gc.collect()
    
    # CatBoost
    if HAS_CATBOOST:
        print("  Training CatBoost...")
        start = time.time()
        cb_model = cb.CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.1,
            random_seed=RANDOM_SEED, verbose=False,
            auto_class_weights='Balanced'
        )
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        cb_time = time.time() - start
        y_prob_cb = cb_model.predict_proba(X_test)[:, 1]
        cb_auc = roc_auc_score(y_test, y_prob_cb)
        
        results['CatBoost'] = {'roc_auc': float(cb_auc), 'time_seconds': float(cb_time)}
        print(f"    AUC: {cb_auc:.3f}, Time: {cb_time:.1f}s")
        
        del cb_model, y_prob_cb
        gc.collect()
    if HAS_TORCH and not skip_slow:
        print("  Training MLP...")
        y_prob_mlp, mlp_time = train_mlp(X_train.copy(), y_train.copy(), 
                                         X_val.copy(), y_val.copy(), X_test.copy())
        mlp_auc = roc_auc_score(y_test, y_prob_mlp)
        
        results['MLP'] = {'roc_auc': float(mlp_auc), 'time_seconds': float(mlp_time)}
        print(f"    AUC: {mlp_auc:.3f}, Time: {mlp_time:.1f}s")
        
        del y_prob_mlp
        gc.collect()
    
    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()
    
    return results

def main():
    skip_slow = '--skip-slow' in sys.argv
    if skip_slow:
        print("NOTE: Skipping slow models")
    
    print("\nLoading data...")
    print_memory("start")
    train_df, val_df, test_df = load_data()
    print_memory("after load")
    
    feature_cols = get_feature_columns()
    
    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '08_dl_comparison.py',
            'latex_table': 'Table 7 (Deep Learning Architecture Comparison)',
            'hardware': 'CPU for tree models, GPU for neural if available',
            'skip_slow': skip_slow
        },
        'tasks': {}
    }
    
    for task_id in TASKS.keys():
        results = compare_models(train_df, val_df, test_df, task_id, feature_cols, skip_slow)
        all_results['tasks'][task_id] = results
        gc.collect()
        print_memory(f"after {task_id}")
    
    output_json = RESULTS_DIR / 'dl_comparison.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_json}")
    
    print("\n" + "=" * 70)
    print("DL COMPARISON COMPLETE")
    del train_df, val_df, test_df
    gc.collect()

if __name__ == "__main__":
    main()
