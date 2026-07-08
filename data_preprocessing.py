#!/usr/bin/env python
# data_preprocessing.py
"""
Complete data preprocessing pipeline for dengue outbreak prediction (fixed).
Changes:
  - Rolling statistics now use shift(1) to avoid data leakage.
  - Dropped redundant columns (IBGE_code, FAO_GAUL_code, etc.).
  - Outbreak_flag redefined as stricter (cases > 100 AND > 1.5*lag1) – optional.
  - Added longer weather lags (up to 12 months).
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ===========================================================================
# CONFIGURATION
# ===========================================================================
RAW_DATA_PATH = "data/filtered_data_SEARO_1769008875181.csv"
OUTPUT_DATA_PATH = "data/merged_data.csv"
WEATHER_CACHE_DIR = "data/weather_cache"

os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
os.makedirs(WEATHER_CACHE_DIR, exist_ok=True)

ISO_TO_LATLON = {
    'BGD': (23.6850, 90.3563),
    'THA': (15.8700, 100.9925),
    'IDN': (-0.7893, 113.9213),
    'IND': (20.5937, 78.9629),
    'LKA': (7.8731, 80.7718),
    'MMR': (21.9162, 95.9560),
    'NPL': (28.3949, 84.1240),
    'MDV': (3.2028, 73.2207),
    'TLS': (-8.8742, 125.7275),
    'BTN': (27.5142, 90.4336),
}

# ===========================================================================
# WEATHER API (unchanged)
# ===========================================================================
def fetch_nasa_power_monthly(lat, lon, start_year, end_year):
    """Fetch monthly temperature and precipitation from NASA POWER API."""
    start_year = max(1981, start_year)
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=T2M,PRECTOTCORR&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_year}&end={end_year}&format=JSON"
    )
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise ConnectionError(f"API request failed: {e}")

    params = data.get('properties', {}).get('parameter', {})
    if 'T2M' not in params:
        raise ValueError(f"Missing T2M. Got: {list(params.keys())}")

    precip_key = 'PRECTOTCORR' if 'PRECTOTCORR' in params else 'PRECTOT'
    t2m_data = params['T2M']
    pr_data = params.get(precip_key, {})

    records = []
    for date_str, temp_val in t2m_data.items():
        if len(date_str) != 6:
            continue
        year = int(date_str[:4])
        month = int(date_str[4:6])
        precip_val = pr_data.get(date_str)
        if precip_val is not None:
            records.append({
                'year': year,
                'month': month,
                'temperature_c': temp_val,
                'precipitation_mm': precip_val
            })
    if not records:
        raise ValueError("No valid weather records extracted.")
    return pd.DataFrame(records)

def add_weather_to_dataframe(df, country_col='ISO_A0'):
    """Add weather columns to the dengue dataframe."""
    cpy = df.copy()
    cpy['calendar_start_date'] = pd.to_datetime(cpy['calendar_start_date'])
    cpy['year'] = cpy['calendar_start_date'].dt.year
    cpy['month'] = cpy['calendar_start_date'].dt.month

    start_year = max(1981, cpy['year'].min())
    end_year = cpy['year'].max()

    weather_frames = []
    for iso in cpy[country_col].unique():
        if iso not in ISO_TO_LATLON:
            print(f"  Skipping {iso}: no lat/lon mapping.")
            continue
        lat, lon = ISO_TO_LATLON[iso]
        cache_file = f"{WEATHER_CACHE_DIR}/{iso}_{start_year}_{end_year}.parquet"
        if os.path.exists(cache_file):
            print(f" Loading cached weather for {iso}")
            w_df = pd.read_parquet(cache_file)
        else:
            print(f"Fetching weather for {iso} ({lat}, {lon})...")
            try:
                w_df = fetch_nasa_power_monthly(lat, lon, start_year, end_year)
                w_df.to_parquet(cache_file, index=False)
                print(f"   ✓ Saved to cache.")
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue
            time.sleep(0.5)
        w_df[country_col] = iso
        weather_frames.append(w_df)

    if not weather_frames:
        print(" No weather data fetched. Returning original dataframe.")
        return cpy

    weather_all = pd.concat(weather_frames, ignore_index=True)
    merged = cpy.merge(weather_all, on=[country_col, 'year', 'month'], how='left')
    print(f" Merged weather. Rows: {len(merged)}, Missing cells: {merged[['temperature_c','precipitation_mm']].isna().sum().sum()}")
    return merged

# ===========================================================================
# EDA (unchanged but we keep it)
# ===========================================================================
def run_eda(df):
    """Generate summary statistics and plots (optional)."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print("\n Basic Info:")
    print(df.info())
    print("\n  Descriptive Statistics:")
    print(df.describe())
    print(f"\n Unique countries: {df['adm_0_name'].unique()}")
    print(f"   ISO codes: {df['ISO_A0'].unique()}")

    total_cases = df['dengue_total'].sum()
    print(f"\n🦟 Total dengue cases (all time): {total_cases:,}")
    print("   By country:")
    print(df.groupby('adm_0_name')['dengue_total'].sum().sort_values(ascending=False))

    # Time series plot
    df['calendar_start_date'] = pd.to_datetime(df['calendar_start_date'])
    agg = df.groupby(['ISO_A0', 'calendar_start_date'])['dengue_total'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    for country, grp in agg.groupby('ISO_A0'):
        ax.plot(grp['calendar_start_date'], grp['dengue_total'], marker='o', label=country, linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Dengue cases')
    ax.set_title('Dengue cases over time by country')
    ax.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('figures/dengue_timeseries.png', dpi=150)
    plt.show()
    print(" Time series plot saved.")

    # Monthly boxplot
    df['month'] = df['calendar_start_date'].dt.month
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x='month', y='dengue_total', ax=ax)
    ax.set_title('Dengue cases by month (seasonality)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Dengue cases')
    plt.tight_layout()
    plt.savefig('figures/dengue_monthly_boxplot.png', dpi=150)
    plt.show()
    print(" Seasonal boxplot saved.")

# ===========================================================================
# FEATURE ENGINEERING (FIXED)
# ===========================================================================
def engineer_features(df):
    """
    Add advanced time-series features with NO DATA LEAKAGE.
    Rolling stats are computed using shift(1) so they only use past values.
    Redundant columns are dropped.
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING (FIXED - NO LEAKAGE)")
    print("="*60)

    df = df.copy()
    df['calendar_start_date'] = pd.to_datetime(df['calendar_start_date'])
    df = df.sort_values(['ISO_A0', 'calendar_start_date']).reset_index(drop=True)

    # ---- Drop redundant columns ----
    cols_to_drop = [
        'adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name',
        'FAO_GAUL_code', 'RNE_iso_code', 'IBGE_code',
        'calendar_end_date', 'Year', 'case_definition_standardised',
        'S_res', 'T_res', 'UUID', 'region'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

    # ---- Basic time features ----
    df['year'] = df['calendar_start_date'].dt.year
    df['month'] = df['calendar_start_date'].dt.month
    df['dayofyear'] = df['calendar_start_date'].dt.dayofyear
    df['weekofyear'] = df['calendar_start_date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['month'].apply(lambda x: (x-1)//3 + 1)
    df['is_rainy_season'] = df['month'].isin([6,7,8,9,10]).astype(int)

    # Cyclical encoding
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['dayofyear_sin'] = np.sin(2*np.pi*df['dayofyear']/365)
    df['dayofyear_cos'] = np.cos(2*np.pi*df['dayofyear']/365)

    # ---- Dengue lags ----
    for lag in [1,2,3,6,12]:
        df[f'dengue_lag_{lag}'] = df.groupby('ISO_A0')['dengue_total'].shift(lag)

    # ---- Rolling statistics (FIXED: shift(1) before rolling) ----
    for window in [3,6,12]:
        # Moving average (only past values)
        df[f'dengue_ma_{window}'] = df.groupby('ISO_A0')['dengue_total'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'dengue_std_{window}'] = df.groupby('ISO_A0')['dengue_total'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # ---- Weather features ----
    weather_cols = ['temperature_c', 'precipitation_mm']
    for col in weather_cols:
        if col not in df.columns:
            continue
        # Lags (1,2,3,6,12)
        for lag in [1,2,3,6,12]:
            df[f'{col}_lag_{lag}'] = df.groupby('ISO_A0')[col].shift(lag)
        # Rolling stats (past only)
        for window in [3,6]:
            df[f'{col}_ma_{window}'] = df.groupby('ISO_A0')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'{col}_std_{window}'] = df.groupby('ISO_A0')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )

    # Interaction: temp * precip (both current values, but this is allowed because it's not lagged)
    if 'temperature_c' in df.columns and 'precipitation_mm' in df.columns:
        df['temp_precip_interaction'] = df['temperature_c'] * df['precipitation_mm']

    # ---- Trend features ----
    df['time_index'] = df.groupby('ISO_A0').cumcount() + 1
    min_year = df['year'].min()
    max_year = df['year'].max()
    df['year_normalized'] = (df['year'] - min_year) / (max_year - min_year + 1)

    # ---- Encode country ----
    le = LabelEncoder()
    df['ISO_A0_encoded'] = le.fit_transform(df['ISO_A0'])

    # ---- Redefine outbreak_flag (optional) ----
    # Stricter definition: cases > 100 AND > 1.5 * previous month's cases
    # Uncomment the following lines if you want this stricter target.
    # df['outbreak_flag'] = ((df['dengue_total'] > 100) &
    #                        (df['dengue_total'] > 1.5 * df['dengue_lag_1'])).astype(int)
    # Otherwise keep original:
    if 'outbreak_flag' not in df.columns:
        df['outbreak_flag'] = (df['dengue_total'] > 0).astype(int)

    print(f" Feature engineering complete. New shape: {df.shape}")
    return df

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("="*60)
    print("DENGUE OUTBREAK DATA PREPROCESSING (FIXED)")
    print("="*60)

    # 1. Load raw data
    print("\n Loading raw dengue data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"File not found: {RAW_DATA_PATH}")
        sys.exit(1)
    df_raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"   Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    # 2. EDA (optional, you can comment out for speed)
    run_eda(df_raw)

    # 3. Merge weather
    print("\n" + "="*60)
    print("WEATHER DATA MERGING")
    print("="*60)
    df_with_weather = add_weather_to_dataframe(df_raw)
    df_with_weather = df_with_weather[df_with_weather['year'] >= 1981].reset_index(drop=True)
    print(f"Rows after filtering to years >= 1981: {len(df_with_weather)}")

    # 4. Feature engineering
    df_final = engineer_features(df_with_weather)

    # 5. Drop rows with NaN (due to lags/rolling)
    initial_len = len(df_final)
    df_final = df_final.dropna().reset_index(drop=True)
    dropped = initial_len - len(df_final)
    print(f"\n Dropped {dropped} rows with NaNs. Remaining: {len(df_final)} rows.")

    # 6. Save
    df_final.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\n Final dataset saved to: {OUTPUT_DATA_PATH}")

    # 7. Summary
    print("\n Final feature list:")
    print(df_final.columns.tolist())
    print(f"\n   Total features: {len(df_final.columns)}")
    print(f"   Final rows: {len(df_final)}")
    print("\n Preprocessing complete! Run 'python train.py' now.")

if __name__ == "__main__":
    main()