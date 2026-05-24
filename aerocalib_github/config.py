#!/usr/bin/env python3
"""Project constants, task definitions, and helper functions."""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import gc
import numpy as np

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TRAIN_YEARS = (2010, 2020)  # 11 years for training
VAL_YEARS = (2021, 2022)  # 2 years for validation/calibration
TEST_YEARS = (2023, 2024)  # 2 years for final evaluation

@dataclass
class TaskConfig:
    name: str
    description: str
    threshold: float
    unit: str
    horizon: int
    comparison: str  # 'lt' for <, 'gt' for >

TASKS = {
    'T1': TaskConfig(
        name='Low Visibility',
        description='Visibility below 1000 meters',
        threshold=1000.0,
        unit='meters',
        horizon=6,
        comparison='lt'
    ),
    'T2': TaskConfig(
        name='Low Ceiling',
        description='Ceiling below 200 feet OR (ceiling unreported AND fog/mist present)',
        threshold=200.0,
        unit='feet',
        horizon=6,
        comparison='lt'
    ),
    'T3': TaskConfig(
        name='Crosswind Exceedance',
        description='Crosswind component exceeding 15 knots',
        threshold=15.0,
        unit='knots',
        horizon=6,
        comparison='gt'
    ),
    'T4': TaskConfig(
        name='Thunderstorm',
        description='TS family tokens present in METAR',
        threshold=0.0,  # Binary - presence of TS tokens
        unit='binary',
        horizon=1,
        comparison='eq'
    ),
}

# TS family tokens for T4 (METAR present weather codes)
TS_TOKENS = ['TS', 'TSRA', '+TSRA', '-TSRA', 'TSGR', 'TSGS', 'TSSN',
             'TSPL', 'TSFG', 'TSDS', 'TSSS', 'VCTS']

# Fog/mist tokens for T2 compound definition
FOG_MIST_TOKENS = ['FG', 'BR', 'MIFG', 'BCFG', 'PRFG', 'FZFG']
# Lag hours for feature engineering
LAG_HOURS = [1, 2, 3, 6, 12, 24]

# Rolling window sizes (hours)
ROLLING_WINDOWS = [6, 24]

# Base meteorological variables for lagging
LAG_VARIABLES = ['wind_speed', 'wind_dir', 'visibility', 'ceiling',
                 'temperature', 'dewpoint', 'pressure']

# Variables actually used in model (excluding high-NaN ones)
LAG_VARIABLES_SAFE = ['wind_speed', 'wind_dir', 'visibility',
                      'temperature', 'dewpoint']

# Binary indicators
BINARY_INDICATORS = ['has_fog', 'has_precip', 'has_gust']
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'early_stopping_rounds': 10,
    'random_state': 42,
}

# Scale pos weight is computed dynamically: min((1-p)/p, 100)
MAX_SCALE_POS_WEIGHT = 100

def optimize_dtypes(df, verbose=False):
    """
    Optimize DataFrame dtypes to reduce memory usage.
    - float64 -> float32
    - int64 -> int32 (where possible)
    - object -> category (for low-cardinality strings)
    """
    import pandas as pd
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Low cardinality
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f"  Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    
    return df

def memory_usage_mb():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def clear_memory():
    gc.collect()

def print_memory(label=""):
    try:
        mem = memory_usage_mb()
        print(f"  [MEM {label}]: {mem:.0f} MB")
    except:
        pass  # psutil not available

def get_feature_columns() -> List[str]:
    """
    Get the list of feature columns used for training.
    EXCLUDES ceiling_lag* and pressure_lag* due to high NaN rates.
    """
    features = []

    # Time encoding features (7)
    features.extend([
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos',
        'dayofweek_sin', 'dayofweek_cos',
        'dayofyear'
    ])

    # Lag features for meteorological variables
    lag_vars = [
        'wind_speed', 'wind_dir', 'wind_u', 'wind_v',
        'wind_dir_sin', 'wind_dir_cos',
        'visibility',
        'temperature',
        'dewpoint',
        'ceiling',
        'pressure',
    ]

    for var in lag_vars:
        for lag in LAG_HOURS:
            features.append(f'{var}_lag{lag}')

    # Binary indicator lags (18)
    for var in ['has_fog', 'has_precip', 'has_gust']:
        for lag in LAG_HOURS:
            features.append(f'{var}_lag{lag}')

    # Rolling statistics (12)
    for var in ['wind_u', 'wind_v', 'wind_speed']:
        for window in ROLLING_WINDOWS:
            features.append(f'{var}_roll{window}_mean')
            features.append(f'{var}_roll{window}_std')

    # Location features (3)
    features.extend(['lat', 'lon', 'rwy_hdg'])

    return features

# Expose as constant for backward compatibility
FEATURE_COLUMNS = get_feature_columns()
EXPECTED_FEATURE_COUNT = len(FEATURE_COLUMNS)

# Climate zones for stratified reporting
CLIMATE_ZONES = {
    'Northern Europe': ['EGLL', 'EHAM', 'LFPG', 'EDDF', 'EKCH', 'ENGM',
                        'EFHK', 'ESSA', 'EGKK', 'EDDM'],
    'Mediterranean': ['LEMD', 'LEBL', 'LIRF', 'LGAV', 'LTFM', 'LTBA'],
    'Continental': ['KJFK', 'KORD', 'KDEN', 'KATL', 'KDFW', 'CYYZ',
                    'UUEE', 'UUDD', 'KLAX', 'KSFO'],
    'East Asian': ['RJTT', 'RJAA', 'RKSI', 'VHHH', 'RCTP', 'ZSSS', 'ZBAA'],
    'Tropical': ['WSSS', 'WMKK', 'VTBS', 'WIII', 'VABB', 'OMDB',
                 'OERK', 'VIDP'],
    'Southern Hemisphere': ['YSSY', 'YMML', 'FAOR', 'SBGR', 'SCEL',
                            'SAEZ', 'NZAA', 'NZCH', 'FACT', 'SBBR'],
}

# Airport metadata with runway headings
AIRPORTS = {
    # Northern Europe
    'EGLL': {'name': 'Heathrow', 'city': 'London', 'country': 'UK', 'lat': 51.47, 'lon': -0.45, 'rwy_hdg': 270, 'zone': 'Northern Europe'},
    'EHAM': {'name': 'Schiphol', 'city': 'Amsterdam', 'country': 'Netherlands', 'lat': 52.31, 'lon': 4.76, 'rwy_hdg': 183, 'zone': 'Northern Europe'},
    'LFPG': {'name': 'Charles de Gaulle', 'city': 'Paris', 'country': 'France', 'lat': 49.01, 'lon': 2.55, 'rwy_hdg': 90, 'zone': 'Northern Europe'},
    'EDDF': {'name': 'Frankfurt', 'city': 'Frankfurt', 'country': 'Germany', 'lat': 50.04, 'lon': 8.56, 'rwy_hdg': 70, 'zone': 'Northern Europe'},
    'EKCH': {'name': 'Kastrup', 'city': 'Copenhagen', 'country': 'Denmark', 'lat': 55.62, 'lon': 12.65, 'rwy_hdg': 40, 'zone': 'Northern Europe'},
    'ENGM': {'name': 'Gardermoen', 'city': 'Oslo', 'country': 'Norway', 'lat': 60.19, 'lon': 11.10, 'rwy_hdg': 19, 'zone': 'Northern Europe'},
    'EFHK': {'name': 'Helsinki-Vantaa', 'city': 'Helsinki', 'country': 'Finland', 'lat': 60.32, 'lon': 24.96, 'rwy_hdg': 40, 'zone': 'Northern Europe'},
    'ESSA': {'name': 'Arlanda', 'city': 'Stockholm', 'country': 'Sweden', 'lat': 59.65, 'lon': 17.92, 'rwy_hdg': 10, 'zone': 'Northern Europe'},
    'EGKK': {'name': 'Gatwick', 'city': 'London', 'country': 'UK', 'lat': 51.15, 'lon': -0.18, 'rwy_hdg': 260, 'zone': 'Northern Europe'},
    'EDDM': {'name': 'Munich', 'city': 'Munich', 'country': 'Germany', 'lat': 48.35, 'lon': 11.79, 'rwy_hdg': 80, 'zone': 'Northern Europe'},
    # Mediterranean
    'LEMD': {'name': 'Barajas', 'city': 'Madrid', 'country': 'Spain', 'lat': 40.50, 'lon': -3.57, 'rwy_hdg': 320, 'zone': 'Mediterranean'},
    'LEBL': {'name': 'El Prat', 'city': 'Barcelona', 'country': 'Spain', 'lat': 41.30, 'lon': 2.08, 'rwy_hdg': 70, 'zone': 'Mediterranean'},
    'LIRF': {'name': 'Fiumicino', 'city': 'Rome', 'country': 'Italy', 'lat': 41.80, 'lon': 12.24, 'rwy_hdg': 160, 'zone': 'Mediterranean'},
    'LGAV': {'name': 'Eleftherios Venizelos', 'city': 'Athens', 'country': 'Greece', 'lat': 37.94, 'lon': 23.94, 'rwy_hdg': 30, 'zone': 'Mediterranean'},
    'LTFM': {'name': 'Istanbul Airport', 'city': 'Istanbul', 'country': 'Turkey', 'lat': 41.26, 'lon': 28.74, 'rwy_hdg': 175, 'zone': 'Mediterranean'},
    'LTBA': {'name': 'Ataturk', 'city': 'Istanbul', 'country': 'Turkey', 'lat': 40.98, 'lon': 28.81, 'rwy_hdg': 52, 'zone': 'Mediterranean'},
    # Continental
    'KJFK': {'name': 'John F. Kennedy', 'city': 'New York', 'country': 'USA', 'lat': 40.64, 'lon': -73.78, 'rwy_hdg': 310, 'zone': 'Continental'},
    'KORD': {'name': "O'Hare", 'city': 'Chicago', 'country': 'USA', 'lat': 41.97, 'lon': -87.91, 'rwy_hdg': 280, 'zone': 'Continental'},
    'KDEN': {'name': 'Denver International', 'city': 'Denver', 'country': 'USA', 'lat': 39.86, 'lon': -104.67, 'rwy_hdg': 350, 'zone': 'Continental'},
    'KATL': {'name': 'Hartsfield-Jackson', 'city': 'Atlanta', 'country': 'USA', 'lat': 33.64, 'lon': -84.43, 'rwy_hdg': 270, 'zone': 'Continental'},
    'KDFW': {'name': 'Dallas/Fort Worth', 'city': 'Dallas', 'country': 'USA', 'lat': 32.90, 'lon': -97.04, 'rwy_hdg': 360, 'zone': 'Continental'},
    'CYYZ': {'name': 'Pearson', 'city': 'Toronto', 'country': 'Canada', 'lat': 43.68, 'lon': -79.62, 'rwy_hdg': 50, 'zone': 'Continental'},
    'UUEE': {'name': 'Sheremetyevo', 'city': 'Moscow', 'country': 'Russia', 'lat': 55.97, 'lon': 37.41, 'rwy_hdg': 60, 'zone': 'Continental'},
    'UUDD': {'name': 'Domodedovo', 'city': 'Moscow', 'country': 'Russia', 'lat': 55.41, 'lon': 37.90, 'rwy_hdg': 320, 'zone': 'Continental'},
    'KLAX': {'name': 'Los Angeles International', 'city': 'Los Angeles', 'country': 'USA', 'lat': 33.94, 'lon': -118.41, 'rwy_hdg': 250, 'zone': 'Continental'},
    'KSFO': {'name': 'San Francisco International', 'city': 'San Francisco', 'country': 'USA', 'lat': 37.62, 'lon': -122.38, 'rwy_hdg': 280, 'zone': 'Continental'},
    # East Asian
    'RJTT': {'name': 'Haneda', 'city': 'Tokyo', 'country': 'Japan', 'lat': 35.55, 'lon': 139.78, 'rwy_hdg': 340, 'zone': 'East Asian'},
    'RJAA': {'name': 'Narita', 'city': 'Tokyo', 'country': 'Japan', 'lat': 35.77, 'lon': 140.39, 'rwy_hdg': 340, 'zone': 'East Asian'},
    'RKSI': {'name': 'Incheon', 'city': 'Seoul', 'country': 'South Korea', 'lat': 37.46, 'lon': 126.44, 'rwy_hdg': 330, 'zone': 'East Asian'},
    'VHHH': {'name': 'Hong Kong International', 'city': 'Hong Kong', 'country': 'China', 'lat': 22.31, 'lon': 113.92, 'rwy_hdg': 70, 'zone': 'East Asian'},
    'RCTP': {'name': 'Taoyuan', 'city': 'Taipei', 'country': 'Taiwan', 'lat': 25.08, 'lon': 121.23, 'rwy_hdg': 50, 'zone': 'East Asian'},
    'ZSSS': {'name': 'Hongqiao', 'city': 'Shanghai', 'country': 'China', 'lat': 31.20, 'lon': 121.34, 'rwy_hdg': 360, 'zone': 'East Asian'},
    'ZBAA': {'name': 'Capital', 'city': 'Beijing', 'country': 'China', 'lat': 40.08, 'lon': 116.60, 'rwy_hdg': 360, 'zone': 'East Asian'},
    # Tropical
    'WSSS': {'name': 'Changi', 'city': 'Singapore', 'country': 'Singapore', 'lat': 1.36, 'lon': 103.99, 'rwy_hdg': 20, 'zone': 'Tropical'},
    'WMKK': {'name': 'Kuala Lumpur International', 'city': 'Kuala Lumpur', 'country': 'Malaysia', 'lat': 2.75, 'lon': 101.71, 'rwy_hdg': 320, 'zone': 'Tropical'},
    'VTBS': {'name': 'Suvarnabhumi', 'city': 'Bangkok', 'country': 'Thailand', 'lat': 13.69, 'lon': 100.75, 'rwy_hdg': 190, 'zone': 'Tropical'},
    'WIII': {'name': 'Soekarno-Hatta', 'city': 'Jakarta', 'country': 'Indonesia', 'lat': -6.13, 'lon': 106.66, 'rwy_hdg': 70, 'zone': 'Tropical'},
    'VABB': {'name': 'Chhatrapati Shivaji', 'city': 'Mumbai', 'country': 'India', 'lat': 19.09, 'lon': 72.87, 'rwy_hdg': 90, 'zone': 'Tropical'},
    'OMDB': {'name': 'Dubai International', 'city': 'Dubai', 'country': 'UAE', 'lat': 25.25, 'lon': 55.37, 'rwy_hdg': 300, 'zone': 'Tropical'},
    'OERK': {'name': 'King Khalid', 'city': 'Riyadh', 'country': 'Saudi Arabia', 'lat': 24.96, 'lon': 46.70, 'rwy_hdg': 330, 'zone': 'Tropical'},
    'VIDP': {'name': 'Indira Gandhi International', 'city': 'Delhi', 'country': 'India', 'lat': 28.56, 'lon': 77.10, 'rwy_hdg': 280, 'zone': 'Tropical'},
    # Southern Hemisphere
    'YSSY': {'name': 'Kingsford Smith', 'city': 'Sydney', 'country': 'Australia', 'lat': -33.95, 'lon': 151.18, 'rwy_hdg': 340, 'zone': 'Southern Hemisphere'},
    'YMML': {'name': 'Melbourne', 'city': 'Melbourne', 'country': 'Australia', 'lat': -37.67, 'lon': 144.84, 'rwy_hdg': 340, 'zone': 'Southern Hemisphere'},
    'FAOR': {'name': 'O.R. Tambo', 'city': 'Johannesburg', 'country': 'South Africa', 'lat': -26.14, 'lon': 28.25, 'rwy_hdg': 30, 'zone': 'Southern Hemisphere'},
    'FACT': {'name': 'Cape Town International', 'city': 'Cape Town', 'country': 'South Africa', 'lat': -33.97, 'lon': 18.60, 'rwy_hdg': 10, 'zone': 'Southern Hemisphere'},
    'SBGR': {'name': 'Guarulhos', 'city': 'Sao Paulo', 'country': 'Brazil', 'lat': -23.44, 'lon': -46.47, 'rwy_hdg': 90, 'zone': 'Southern Hemisphere'},
    'SBBR': {'name': 'Brasilia International', 'city': 'Brasilia', 'country': 'Brazil', 'lat': -15.87, 'lon': -47.92, 'rwy_hdg': 110, 'zone': 'Southern Hemisphere'},
    'SCEL': {'name': 'Arturo Merino Benitez', 'city': 'Santiago', 'country': 'Chile', 'lat': -33.39, 'lon': -70.79, 'rwy_hdg': 170, 'zone': 'Southern Hemisphere'},
    'SAEZ': {'name': 'Ezeiza', 'city': 'Buenos Aires', 'country': 'Argentina', 'lat': -34.82, 'lon': -58.54, 'rwy_hdg': 110, 'zone': 'Southern Hemisphere'},
    'NZAA': {'name': 'Auckland', 'city': 'Auckland', 'country': 'New Zealand', 'lat': -37.01, 'lon': 174.79, 'rwy_hdg': 50, 'zone': 'Southern Hemisphere'},
    'NZCH': {'name': 'Christchurch', 'city': 'Christchurch', 'country': 'New Zealand', 'lat': -43.49, 'lon': 172.53, 'rwy_hdg': 20, 'zone': 'Southern Hemisphere'},
}

# Airports to hold out for spatial generalization (20%)
HELDOUT_AIRPORTS = ['EDDF', 'ESSA', 'LGAV', 'OERK', 'KORD', 'RCTP', 'VABB', 'VTBS', 'WSSS']
# Block bootstrap settings
BOOTSTRAP_BLOCK_SIZE = 24  # hours (station-day blocks)
BOOTSTRAP_N_RESAMPLES = 300
BOOTSTRAP_CONFIDENCE = 0.95

# ECE binning
ECE_N_BINS = 10
ECE_STRATEGY = 'quantile'  # Equal-frequency binning

# Cost ratios for threshold optimization (FN:FP)
COST_RATIOS = [10, 25, 50]

# Default cost ratio
DEFAULT_COST_RATIO = 25

# Random seed for reproducibility
RANDOM_SEED = 42

OUTPUT_FILES = {
    'data_accounting': {
        'json': RESULTS_DIR / 'data_accounting.json',
        'latex_table': 'Table 1 (Data Accounting)',
        'script': '02_preprocess_data.py'
    },
    'baselines': {
        'json': RESULTS_DIR / 'baselines.json',
        'latex_table': 'Table 2 (Extended Baseline Comparison)',
        'script': '04_baselines.py'
    },
    'main_results': {
        'json': RESULTS_DIR / 'main_results.json',
        'latex_table': 'Table 3 (Test Set Performance)',
        'script': '03_train_evaluate.py'
    },
    'calibration': {
        'json': RESULTS_DIR / 'calibration.json',
        'latex_table': 'Table 4 (Calibration Summary)',
        'script': '05_calibration.py'
    },
    'airport_heldout': {
        'json': RESULTS_DIR / 'airport_heldout.json',
        'latex_table': 'Table 5 (Airport-Held-Out Generalization)',
        'script': '06_airport_heldout.py'
    },
    't2_ablation': {
        'json': RESULTS_DIR / 't2_ablation.json',
        'latex_table': 'Table 6 (T2 Leakage Checks)',
        'script': '07_t2_ablation.py'
    },
    'dl_comparison': {
        'json': RESULTS_DIR / 'dl_comparison.json',
        'latex_table': 'Table 7 (Deep Learning Architecture Comparison)',
        'script': '08_dl_comparison.py'
    },
    'cost_benefit': {
        'json': RESULTS_DIR / 'cost_benefit.json',
        'latex_table': 'Table 8 (Cost-Optimal Thresholds)',
        'script': '09_cost_benefit.py'
    },
    'transfer': {
        'json': RESULTS_DIR / 'transfer.json',
        'latex_table': 'Table 9 (Transfer Learning)',
        'script': '10_transfer_learning.py'
    },
    'nwp_proxy': {
        'json': RESULTS_DIR / 'nwp_proxy.json',
        'latex_table': 'Table 10 (NWP Proxy Comparison)',
        'script': '11_nwp_proxy.py'
    },
}

def get_airport_zone(icao: str) -> str:
    return AIRPORTS.get(icao, {}).get('zone', 'Unknown')

def get_zone_airports(zone: str) -> List[str]:
    return CLIMATE_ZONES.get(zone, [])

def get_all_airports() -> List[str]:
    return list(AIRPORTS.keys())

def compute_scale_pos_weight(prevalence: float) -> float:
    if prevalence <= 0 or prevalence >= 1:
        return 1.0
    return min((1 - prevalence) / prevalence, MAX_SCALE_POS_WEIGHT)

def get_lag_only_features(task_id: str) -> List[str]:
    features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos']

    if task_id == 'T1':
        for lag in LAG_HOURS:
            features.append(f'visibility_lag{lag}')
    elif task_id == 'T2':
        for lag in LAG_HOURS:
            features.append(f'visibility_lag{lag}')
            features.append(f'has_fog_lag{lag}')
    elif task_id == 'T3':
        for lag in LAG_HOURS:
            features.append(f'wind_speed_lag{lag}')
            features.append(f'wind_dir_lag{lag}')
    elif task_id == 'T4':
        for lag in LAG_HOURS:
            features.append(f'has_fog_lag{lag}')
            features.append(f'has_precip_lag{lag}')

    return features

if __name__ == '__main__':
    print(f"Total airports: {len(AIRPORTS)}")
    print(f"Climate zones: {len(CLIMATE_ZONES)}")
    print(f"Held-out airports: {len(HELDOUT_AIRPORTS)}")
    print(f"\nFeature columns: {len(FEATURE_COLUMNS)}")
    print(f"\nTasks:")
    for task_id, task in TASKS.items():
        print(f"  {task_id}: {task.name} (h={task.horizon})")
