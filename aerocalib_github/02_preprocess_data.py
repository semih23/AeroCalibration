#!/usr/bin/env python3
"""Processes raw METAR data and creates labels for all tasks."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
import re
import gc

warnings.filterwarnings('ignore')

from config import (
    AIRPORTS, RAW_DIR, PROCESSED_DIR, RESULTS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    TS_TOKENS, FOG_MIST_TOKENS, LAG_HOURS, ROLLING_WINDOWS,
    RANDOM_SEED, optimize_dtypes, clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

def extract_visibility_meters(df: pd.DataFrame) -> pd.Series:
    vsby = pd.to_numeric(df['vsby'], errors='coerce')
    vis_meters = (vsby * 1609.34).astype('float32')
    vis_meters = vis_meters.clip(upper=9999)
    return vis_meters

def extract_t1_label(df: pd.DataFrame) -> pd.Series:
    vis_meters = extract_visibility_meters(df)
    threshold = TASKS['T1'].threshold
    return (vis_meters < threshold).astype('int8')

def extract_ceiling_feet(df: pd.DataFrame) -> pd.Series:
    ceiling = pd.Series(np.nan, index=df.index, dtype='float32')
    
    for i in range(1, 5):
        skyc_col = f'skyc{i}'
        skyl_col = f'skyl{i}'
        
        if skyc_col in df.columns and skyl_col in df.columns:
            is_ceiling = df[skyc_col].isin(['BKN', 'OVC', 'VV'])
            height = pd.to_numeric(df[skyl_col], errors='coerce').astype('float32')
            
            valid_ceiling = is_ceiling & height.notna()
            ceiling = ceiling.where(~valid_ceiling, np.minimum(ceiling.fillna(np.inf), height))
    
    ceiling = ceiling.replace([np.inf], np.nan)
    return ceiling

def has_fog_mist(df: pd.DataFrame) -> pd.Series:
    if 'wxcodes' not in df.columns:
        return pd.Series(False, index=df.index)
    
    wx = df['wxcodes'].fillna('').astype(str).str.upper()
    pattern = '|'.join(FOG_MIST_TOKENS)
    return wx.str.contains(pattern, na=False, regex=True)

def extract_t2_label(df: pd.DataFrame) -> pd.Series:
    ceiling = extract_ceiling_feet(df)
    threshold = TASKS['T2'].threshold
    
    low_ceiling = (ceiling < threshold) & ceiling.notna()
    ceiling_missing = ceiling.isna()
    fog_present = has_fog_mist(df)
    compound_condition = ceiling_missing & fog_present
    
    return (low_ceiling | compound_condition).astype('int8')

def compute_crosswind(df: pd.DataFrame, rwy_hdg: float) -> pd.Series:
    drct = pd.to_numeric(df['drct'], errors='coerce').astype('float32')
    sknt = pd.to_numeric(df['sknt'], errors='coerce').astype('float32')
    
    delta = drct - rwy_hdg
    delta_rad = np.radians(delta)
    crosswind = np.abs(sknt * np.sin(delta_rad))
    
    if 'gust' in df.columns:
        gust = pd.to_numeric(df['gust'], errors='coerce').astype('float32')
        gust_crosswind = np.abs(gust * np.sin(delta_rad))
        crosswind = np.maximum(crosswind, gust_crosswind.fillna(0))
    
    return crosswind.astype('float32')

def extract_t3_label(df: pd.DataFrame, rwy_hdg: float) -> pd.Series:
    crosswind = compute_crosswind(df, rwy_hdg)
    threshold = TASKS['T3'].threshold
    return (crosswind > threshold).astype('int8')

def has_thunderstorm(df: pd.DataFrame) -> pd.Series:
    if 'wxcodes' not in df.columns:
        return pd.Series(False, index=df.index)
    
    wx = df['wxcodes'].fillna('').astype(str).str.upper()
    
    has_ts = pd.Series(False, index=df.index)
    for token in TS_TOKENS:
        pattern = r'\b' + re.escape(token) + r'\b'
        has_ts = has_ts | wx.str.contains(pattern, na=False, regex=True)
    
    return has_ts

def extract_t4_label(df: pd.DataFrame) -> pd.Series:
    return has_thunderstorm(df).astype('int8')
# PAST-ONLY HOURLY ALIGNMENT

def align_to_hourly_pastonly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['valid'] = pd.to_datetime(df['valid'], errors='coerce')
    df = df.dropna(subset=['valid'])
    df = df.sort_values('valid')
    
    min_time = df['valid'].min().floor('H')
    max_time = df['valid'].max().ceil('H')
    hourly_index = pd.date_range(min_time, max_time, freq='H')
    
    df = df.set_index('valid')
    df = df[~df.index.duplicated(keep='last')]
    
    hourly_df = df.reindex(hourly_index, method='ffill', tolerance='30min')
    hourly_df = hourly_df.dropna(how='all')
    hourly_df = hourly_df.reset_index()
    hourly_df = hourly_df.rename(columns={'index': 'datetime'})
    
    return hourly_df

def create_features(df: pd.DataFrame, icao: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Use float32 throughout
    df['wind_speed'] = pd.to_numeric(df['sknt'], errors='coerce').astype('float32')
    df['wind_dir'] = pd.to_numeric(df['drct'], errors='coerce').astype('float32')
    df['visibility'] = extract_visibility_meters(df)
    df['ceiling'] = extract_ceiling_feet(df)
    df['temperature'] = pd.to_numeric(df['tmpf'], errors='coerce').astype('float32')
    df['dewpoint'] = pd.to_numeric(df['dwpf'], errors='coerce').astype('float32')
    df['pressure'] = pd.to_numeric(df['mslp'], errors='coerce').astype('float32')
    
    theta = np.radians(df['wind_dir'])
    df['wind_u'] = (-df['wind_speed'] * np.sin(theta)).astype('float32')
    df['wind_v'] = (-df['wind_speed'] * np.cos(theta)).astype('float32')
    
    df['wind_dir_sin'] = np.sin(theta).astype('float32')
    df['wind_dir_cos'] = np.cos(theta).astype('float32')
    
    # Time features - use smaller int types
    df['hour'] = df['datetime'].dt.hour.astype('int8')
    df['month'] = df['datetime'].dt.month.astype('int8')
    df['dayofweek'] = df['datetime'].dt.dayofweek.astype('int8')
    df['dayofyear'] = df['datetime'].dt.dayofyear.astype('int16')
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype('float32')
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7).astype('float32')
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7).astype('float32')
    
    # Binary indicators
    if 'wxcodes' in df.columns:
        wx = df['wxcodes'].fillna('').astype(str).str.upper()
        df['has_fog'] = wx.str.contains('FG|BR', na=False, regex=True).astype('int8')
        df['has_precip'] = wx.str.contains('RA|SN|DZ|SH|PL|GR', na=False, regex=True).astype('int8')
    else:
        df['has_fog'] = np.int8(0)
        df['has_precip'] = np.int8(0)
    
    df['has_gust'] = (pd.to_numeric(df.get('gust', 0), errors='coerce') > 0).astype('int8')
    
    # Lag features
    lag_vars = ['wind_speed', 'wind_dir', 'wind_u', 'wind_v', 'wind_dir_sin', 'wind_dir_cos',
                'visibility', 'ceiling', 'temperature', 'dewpoint', 'pressure']
    
    for var in lag_vars:
        if var in df.columns:
            for lag in LAG_HOURS:
                df[f'{var}_lag{lag}'] = df[var].shift(lag)
    
    for var in ['has_fog', 'has_precip', 'has_gust']:
        for lag in LAG_HOURS:
            df[f'{var}_lag{lag}'] = df[var].shift(lag).astype('float32')  # NaN possible after shift
    
    # Rolling features
    for var in ['wind_u', 'wind_v', 'wind_speed']:
        if var in df.columns:
            for window in ROLLING_WINDOWS:
                df[f'{var}_roll{window}_mean'] = df[var].shift(1).rolling(window, min_periods=1).mean().astype('float32')
                df[f'{var}_roll{window}_std'] = df[var].shift(1).rolling(window, min_periods=1).std().astype('float32')
    
    # Location features - use float32
    rwy_hdg = AIRPORTS[icao]['rwy_hdg']
    df['lat'] = np.float32(AIRPORTS[icao]['lat'])
    df['lon'] = np.float32(AIRPORTS[icao]['lon'])
    df['rwy_hdg'] = np.float32(rwy_hdg)
    
    df['airport'] = icao
    df['climate_zone'] = AIRPORTS[icao]['zone']
    
    return df

def process_airport(icao: str) -> tuple:
    filepath = RAW_DIR / f"metar_{icao}.csv"
    
    if not filepath.exists():
        return icao, None, {'error': 'File not found'}
    
    try:
        # Read with low_memory and specify dtypes where possible
        df = pd.read_csv(filepath, low_memory=False, comment='#')
        initial_count = len(df)
        
        if initial_count < 1000:
            return icao, None, {'error': f'Too few records: {initial_count}'}
        
        df['valid'] = pd.to_datetime(df['valid'], errors='coerce')
        df = df.dropna(subset=['valid'])

        numeric_cols = ['drct', 'sknt', 'gust', 'vsby', 'tmpf', 'dwpf', 'mslp', 
                        'p01i', 'alti', 'relh', 'feel', 'skyl1', 'skyl2', 'skyl3', 'skyl4']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter wind
        valid_wind = (
            df['drct'].notna() & 
            df['sknt'].notna() &
            (df['drct'] >= 0) & (df['drct'] <= 360) &
            (df['sknt'] >= 0) & (df['sknt'] < 150)
        )
        calm_mask = (df['sknt'] == 0)
        df.loc[calm_mask, 'drct'] = 0
        df = df[valid_wind | calm_mask].copy()
        
        # Align to hourly
        hourly_df = align_to_hourly_pastonly(df)
        del df  # Free memory immediately
        gc.collect()
        
        # Create features
        hourly_df = create_features(hourly_df, icao)
        
        # Create labels
        rwy_hdg = AIRPORTS[icao]['rwy_hdg']
        hourly_df['T1_current'] = extract_t1_label(hourly_df)
        hourly_df['T2_current'] = extract_t2_label(hourly_df)
        hourly_df['T3_current'] = extract_t3_label(hourly_df, rwy_hdg)
        hourly_df['T4_current'] = extract_t4_label(hourly_df)
        
        # Future labels
        for task_id, task_config in TASKS.items():
            h = task_config.horizon
            hourly_df[f'{task_id}_h{h}'] = hourly_df[f'{task_id}_current'].shift(-h)
        
        # Drop raw columns to save memory
        raw_cols = ['station', 'metar', 'wxcodes', 'skyc1', 'skyc2', 'skyc3', 'skyc4',
                    'skyl1', 'skyl2', 'skyl3', 'skyl4', 'tmpf', 'dwpf', 'sknt', 'drct',
                    'gust', 'vsby', 'mslp', 'alti', 'p01i', 'relh', 'feel',
                    'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr',
                    'peak_wind_gust', 'peak_wind_drct', 'peak_wind_time',
                    'report_type', 'presentwx']
        for col in raw_cols:
            if col in hourly_df.columns:
                del hourly_df[col]
        
        # Optimize dtypes
        hourly_df = optimize_dtypes(hourly_df, verbose=False)
        
        stats = {
            'initial_records': initial_count,
            'hourly_records': len(hourly_df),
            'date_range': {
                'start': str(hourly_df['datetime'].min()),
                'end': str(hourly_df['datetime'].max())
            },
            'task_prevalence': {
                'T1': float(hourly_df['T1_current'].mean()),
                'T2': float(hourly_df['T2_current'].mean()),
                'T3': float(hourly_df['T3_current'].mean()),
                'T4': float(hourly_df['T4_current'].mean()),
            }
        }
        
        return icao, hourly_df, stats
        
    except Exception as e:
        return icao, None, {'error': str(e)}

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    # Process in batches and save incrementally
    BATCH_SIZE = 10
    airport_list = list(AIRPORTS.keys())
    total = len(airport_list)
    
    all_data = []
    
    for idx, icao in enumerate(airport_list, 1):
        print(f"[{idx}/{total}] Processing {icao}...", end=' ')
        print_memory(f"before {icao}")
        
        icao, df, stats = process_airport(icao)
        
        if df is not None:
            all_data.append(df)
            all_stats[icao] = stats
            print(f"OK ({stats['hourly_records']:,} hourly records)")
        else:
            all_stats[icao] = stats
            print(f"FAILED: {stats.get('error', 'Unknown error')}")
        
        # Periodic garbage collection
        if idx % BATCH_SIZE == 0:
            gc.collect()
            print_memory(f"after batch {idx // BATCH_SIZE}")
    
    if not all_data:
        print("\nERROR: No data processed!")
        return
    
    print("\nCombining all airports...")
    print_memory("before concat")
    
    # Concatenate in batches to avoid memory spike
    CONCAT_BATCH = 10
    combined_parts = []
    
    for i in range(0, len(all_data), CONCAT_BATCH):
        batch = all_data[i:i+CONCAT_BATCH]
        combined_parts.append(pd.concat(batch, ignore_index=True))
        # Clear the batch
        for df in batch:
            del df
        gc.collect()
    
    combined_df = pd.concat(combined_parts, ignore_index=True)
    del combined_parts
    del all_data
    gc.collect()
    
    print_memory("after concat")
    
    # Final dtype optimization
    combined_df = optimize_dtypes(combined_df, verbose=True)
    
    combined_df['year'] = combined_df['datetime'].dt.year.astype('int16')
    
    # Split masks
    train_mask = (combined_df['year'] >= TRAIN_YEARS[0]) & (combined_df['year'] <= TRAIN_YEARS[1])
    val_mask = (combined_df['year'] >= VAL_YEARS[0]) & (combined_df['year'] <= VAL_YEARS[1])
    test_mask = (combined_df['year'] >= TEST_YEARS[0]) & (combined_df['year'] <= TEST_YEARS[1])
    
    # Get feature columns (for accounting)
    exclude_patterns = ['T1_', 'T2_', 'T3_', 'T4_', 'datetime', 'airport', 
                       'climate_zone', 'year']
    feature_cols = [c for c in combined_df.columns 
                    if not any(p in c for p in exclude_patterns)]
    
    # Data accounting
    data_accounting = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '02_preprocess_data.py',
            'latex_table': 'Table 1 (Data Accounting)'
        },
        'summary': {
            'total_airports': len([s for s in all_stats.values() if 'error' not in s]),
            'total_records': len(combined_df),
            'n_features': len(feature_cols),
        },
        'splits': {
            'train': {
                'years': f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
                'n_rows': int(train_mask.sum()),
                'n_stations': int(combined_df.loc[train_mask, 'airport'].nunique())
            },
            'validation': {
                'years': f"{VAL_YEARS[0]}-{VAL_YEARS[1]}",
                'n_rows': int(val_mask.sum()),
                'n_stations': int(combined_df.loc[val_mask, 'airport'].nunique())
            },
            'test': {
                'years': f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
                'n_rows': int(test_mask.sum()),
                'n_stations': int(combined_df.loc[test_mask, 'airport'].nunique())
            }
        },
        'task_test_prevalence': {},
        'airport_stats': all_stats
    }
    
    # Task prevalence on test set
    test_df = combined_df[test_mask]
    for task_id, task_config in TASKS.items():
        h = task_config.horizon
        label_col = f'{task_id}_h{h}'
        if label_col in test_df.columns:
            test_labels = test_df[label_col].dropna()
            n_positive = int(test_labels.sum())
            n_total = len(test_labels)
            prevalence = float(test_labels.mean()) if n_total > 0 else 0.0
            
            data_accounting['task_test_prevalence'][task_id] = {
                'horizon': h,
                'n_positive': n_positive,
                'n_total': n_total,
                'prevalence': prevalence,
                'prevalence_pct': f"{prevalence * 100:.3f}%"
            }
    del test_df
    gc.collect()
    
    # Save parquet with compression
    output_parquet = PROCESSED_DIR / 'aerocalib_hourly.parquet'
    combined_df.to_parquet(output_parquet, index=False, compression='snappy')
    print(f"\nSaved: {output_parquet}")
    
    # Save JSON
    output_json = RESULTS_DIR / 'data_accounting.json'
    with open(output_json, 'w') as f:
        json.dump(data_accounting, f, indent=2, default=str)
    print(f"Saved: {output_json}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATA ACCOUNTING SUMMARY")
    print(f"{'Split':<15} {'Years':<12} {'N Rows':>12} {'N Stations':>12}")
    for split_name, split_data in data_accounting['splits'].items():
        print(f"{split_name.capitalize():<15} {split_data['years']:<12} {split_data['n_rows']:>12,} {split_data['n_stations']:>12}")
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    # Final cleanup
    del combined_df
    gc.collect()
    print_memory("final")

if __name__ == "__main__":
    main()
