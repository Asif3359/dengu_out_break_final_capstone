#!/usr/bin/env python
# eda_analysis.py
"""
Comprehensive exploratory data analysis for:
1. Raw dengue dataset (before weather merge)
2. Final merged dataset (after weather and feature engineering)
3. Effect of SMOTE on class balance (demonstration)

This script generates summary statistics, class balance by country,
correlation heatmaps, feature importance, time-series plots, and seasonality.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ===========================================================================
# CONFIGURATION
# ===========================================================================
RAW_DATA_PATH = "data/filtered_data_SEARO_1769008875181.csv"
MERGED_DATA_PATH = "data/merged_data.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# ===========================================================================
# 1. ANALYSIS OF RAW DENGUE DATASET
# ===========================================================================
def analyze_raw_data():
    print("\n" + "="*70)
    print(" ANALYSIS OF RAW DENGUE DATASET (BEFORE WEATHER MERGE)")
    print("="*70)

    df_raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"\n✅ Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns.")

    print("\n📊 Basic Info:")
    print(df_raw.info())

    print("\n📈 Descriptive Statistics:")
    print(df_raw.describe(include='all'))

    print(f"\n🌍 Unique countries: {df_raw['adm_0_name'].unique()}")
    print(f"   ISO codes: {df_raw['ISO_A0'].unique()}")

    total_cases = df_raw['dengue_total'].sum()
    print(f"\n🦟 Total dengue cases (all time): {total_cases:,}")
    print("   By country:")
    print(df_raw.groupby('adm_0_name')['dengue_total'].sum().sort_values(ascending=False))

    # Missing values
    print("\n🔍 Missing values per column:")
    print(df_raw.isnull().sum())

    # ========== Raw: Class distribution per country ==========
    print("\n" + "-"*70)
    print(" CLASS DISTRIBUTION BY COUNTRY (RAW DATA)")
    print("-"*70)
    df_raw['outbreak_flag'] = (df_raw['dengue_total'] > 0).astype(int)
    raw_country_counts = df_raw.groupby('ISO_A0')['outbreak_flag'].value_counts().unstack(fill_value=0)
    raw_country_counts.columns = ['No Outbreak', 'Outbreak']
    raw_country_counts['Total'] = raw_country_counts.sum(axis=1)
    raw_country_counts['Outbreak %'] = (raw_country_counts['Outbreak'] / raw_country_counts['Total'] * 100).round(2)
    print("\n📊 Detailed class distribution by country (raw):")
    print(raw_country_counts.to_string())

    # Stacked bar for raw
    fig, ax = plt.subplots(figsize=(12, 6))
    raw_country_counts[['No Outbreak', 'Outbreak']].plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'coral'])
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of Months')
    ax.set_title('Outbreak vs Non-Outbreak Counts by Country (Raw Data)')
    ax.legend(title='Class', labels=['No Outbreak', 'Outbreak'])
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/raw_class_counts_by_country.png", dpi=150)
    plt.show()
    print(f"✅ Saved raw class counts bar chart to {FIGURES_DIR}/raw_class_counts_by_country.png")

    # Time series plot (raw data)
    df_raw['calendar_start_date'] = pd.to_datetime(df_raw['calendar_start_date'])
    agg = df_raw.groupby(['ISO_A0', 'calendar_start_date'])['dengue_total'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    for country, grp in agg.groupby('ISO_A0'):
        ax.plot(grp['calendar_start_date'], grp['dengue_total'], label=country, linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Dengue cases')
    ax.set_title('Raw Dengue Cases Over Time by Country (Before Weather Merge)')
    ax.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/raw_dengue_timeseries.png", dpi=150)
    plt.show()
    print(f"✅ Raw time-series plot saved to {FIGURES_DIR}/raw_dengue_timeseries.png")

    return df_raw

# ===========================================================================
# 2. ANALYSIS OF MERGED DATASET
# ===========================================================================
def analyze_merged_data():
    print("\n" + "="*70)
    print(" ANALYSIS OF FINAL MERGED DATASET (WITH WEATHER & FEATURES)")
    print("="*70)

    df = pd.read_csv(MERGED_DATA_PATH, low_memory=False)
    print(f"\n✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Basic info & missing values
    print("\n📊 Basic Info:")
    print(df.info())

    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())

    # Target variable
    df['outbreak_flag'] = (df['dengue_total'] > 0).astype(int)

    print("\n📊 Summary of 'dengue_total':")
    print(df['dengue_total'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    outbreak_counts = df['outbreak_flag'].value_counts()
    print("\n🦟 Outbreak vs Non-Outbreak counts:")
    print(outbreak_counts)
    print(f"   Percentage of outbreaks: {outbreak_counts[1] / len(df) * 100:.2f}%")

    # ========== Merged: Class distribution per country ==========
    print("\n" + "-"*70)
    print(" CLASS DISTRIBUTION BY COUNTRY (MERGED DATA)")
    print("-"*70)
    country_counts = df.groupby('ISO_A0')['outbreak_flag'].value_counts().unstack(fill_value=0)
    country_counts.columns = ['No Outbreak', 'Outbreak']
    country_counts['Total'] = country_counts.sum(axis=1)
    country_counts['Outbreak %'] = (country_counts['Outbreak'] / country_counts['Total'] * 100).round(2)
    print("\n📊 Detailed class distribution by country (merged):")
    print(country_counts.to_string())

    # Stacked bar chart for merged
    fig, ax = plt.subplots(figsize=(12, 6))
    country_counts[['No Outbreak', 'Outbreak']].plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'coral'])
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of Months')
    ax.set_title('Outbreak vs Non-Outbreak Counts by Country (Merged)')
    ax.legend(title='Class', labels=['No Outbreak', 'Outbreak'])
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/merged_class_counts_by_country.png", dpi=150)
    plt.show()
    print(f"✅ Saved merged class counts bar chart to {FIGURES_DIR}/merged_class_counts_by_country.png")

    # Target distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['dengue_total'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Dengue cases')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Dengue Cases (Merged)')
    axes[0].grid(True, alpha=0.3)

    sns.boxplot(data=df, x='ISO_A0', y='dengue_total', ax=axes[1])
    axes[1].set_xlabel('Country')
    axes[1].set_ylabel('Dengue cases')
    axes[1].set_title('Dengue Cases by Country (Merged)')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/merged_target_distribution.png", dpi=150)
    plt.show()
    print(f"✅ Saved target distribution plot to {FIGURES_DIR}/merged_target_distribution.png")

    # Numerical feature summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['dengue_total', 'outbreak_flag', 'Year', 'time_index', 'year_normalized']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"\n📈 Summary of {len(feature_cols)} numeric features (showing first 5):")
    print(df[feature_cols].describe().T.head(10))

    # Outlier detection
    print("\n🔍 Outlier detection (IQR method) for top 5 features:")
    for col in feature_cols[:5]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"   {col}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")

    # Correlation matrix
    subset_cols = [
        'dengue_lag_1', 'dengue_lag_2', 'dengue_lag_3', 'dengue_lag_6', 'dengue_lag_12',
        'dengue_ma_3', 'dengue_ma_6', 'dengue_ma_12',
        'temperature_c', 'temperature_c_lag_1', 'temperature_c_lag_2', 'temperature_c_lag_3',
        'precipitation_mm', 'precipitation_mm_lag_1', 'precipitation_mm_lag_2', 'precipitation_mm_lag_3',
        'temp_precip_interaction',
        'month_sin', 'month_cos', 'is_rainy_season'
    ]
    subset_cols = [col for col in subset_cols if col in df.columns]
    corr_matrix = df[subset_cols + ['dengue_total']].corr()
    target_corr = corr_matrix['dengue_total'].sort_values(ascending=False)
    print("\n🔗 Top 10 features correlated with dengue_total (absolute value):")
    print(target_corr.abs().sort_values(ascending=False).head(10))

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Key Features (Merged)')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/merged_correlation_heatmap.png", dpi=150)
    plt.show()
    print(f"✅ Saved correlation heatmap to {FIGURES_DIR}/merged_correlation_heatmap.png")

    # Feature importance (Random Forest Regressor)
    X = df[feature_cols]
    y = df['dengue_total']
    print(f"\nTraining Random Forest on {len(X)} rows with {len(feature_cols)} features...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_reg.fit(X, y)
    importances = rf_reg.feature_importances_
    feature_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    print("\n🏆 Top 10 most important features (regression):")
    print(feature_imp.head(10))

    plt.figure(figsize=(10, 8))
    feature_imp.head(15).plot(kind='barh', color='teal')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance (Random Forest Regressor) - Merged Data')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/merged_feature_importance.png", dpi=150)
    plt.show()
    print(f"✅ Saved feature importance plot to {FIGURES_DIR}/merged_feature_importance.png")

    # Time series by country (merged)
    df['calendar_start_date'] = pd.to_datetime(df['calendar_start_date'])
    countries = df['ISO_A0'].unique()
    fig, axes = plt.subplots(nrows=len(countries), ncols=1, figsize=(14, 3*len(countries)))
    if len(countries) == 1:
        axes = [axes]
    for ax, country in zip(axes, countries):
        sub = df[df['ISO_A0'] == country].sort_values('calendar_start_date')
        ax.plot(sub['calendar_start_date'], sub['dengue_total'], label=country, linewidth=1.5)
        ax.set_title(f'{country} - Dengue Cases Over Time (Merged)')
        ax.set_ylabel('Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/merged_timeseries_by_country.png", dpi=150)
    plt.show()
    print(f"✅ Saved time series plots to {FIGURES_DIR}/merged_timeseries_by_country.png")

    # Seasonality
    monthly_avg = df.groupby('month')['dengue_total'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_avg['month'], monthly_avg['dengue_total'], color='coral')
    plt.xlabel('Month')
    plt.ylabel('Average Dengue Cases')
    plt.title('Average Dengue Cases by Month (all countries) - Merged')
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/merged_seasonality_monthly.png", dpi=150)
    plt.show()
    print(f"✅ Saved seasonality plot to {FIGURES_DIR}/merged_seasonality_monthly.png")

    # Class imbalance by country (mean)
    country_outbreak = df.groupby('ISO_A0')['outbreak_flag'].mean().sort_values(ascending=False)
    print("\n📊 Outbreak proportion by country (mean):")
    print(country_outbreak)

    plt.figure(figsize=(10, 5))
    country_outbreak.plot(kind='bar', color='skyblue')
    plt.xlabel('Country')
    plt.ylabel('Proportion of Outbreaks')
    plt.title('Outbreak Proportion by Country (Merged)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/merged_outbreak_proportion.png", dpi=150)
    plt.show()
    print(f"✅ Saved outbreak proportion plot to {FIGURES_DIR}/merged_outbreak_proportion.png")

    # Summary
    print("\n" + "="*70)
    print(" MERGED DATASET EDA COMPLETE")
    print("="*70)
    print(f"✅ Total rows: {len(df)}")
    print(f"✅ Total features: {len(df.columns)}")
    print(f"✅ Outbreak percentage: {outbreak_counts[1] / len(df) * 100:.2f}%")
    print(f"✅ Top 3 important features (regression):")
    print(feature_imp.head(3))

    return df, feature_cols

# ===========================================================================
# 3. SMOTE DEMONSTRATION
# ===========================================================================
def demonstrate_smote(df, feature_cols):
    print("\n" + "="*70)
    print(" SMOTE APPLICATION ON MERGED DATA (DEMONSTRATION)")
    print("="*70)

    # Use the same features as in training (excluding target and non-features)
    target = 'outbreak_flag'
    X = df[feature_cols]
    y = df[target]

    print(f"Original class distribution: {np.bincount(y)}")

    # Split into train and test (temporary)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train class distribution (before SMOTE): {np.bincount(y_train)}")

    # Apply SMOTE only on training
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Train class distribution (after SMOTE): {np.bincount(y_train_res)}")

    # Show as a DataFrame
    before = pd.DataFrame({'Class': ['No Outbreak', 'Outbreak'], 'Count': np.bincount(y_train)})
    after = pd.DataFrame({'Class': ['No Outbreak', 'Outbreak'], 'Count': np.bincount(y_train_res)})
    print("\n📊 Class balance comparison (training set):")
    print("Before SMOTE:")
    print(before)
    print("\nAfter SMOTE:")
    print(after)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=before, x='Class', y='Count', ax=axes[0], palette='coolwarm')
    axes[0].set_title('Before SMOTE')
    axes[0].set_ylabel('Number of samples')
    sns.barplot(data=after, x='Class', y='Count', ax=axes[1], palette='coolwarm')
    axes[1].set_title('After SMOTE')
    axes[1].set_ylabel('Number of samples')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/smote_effect.png", dpi=150)
    plt.show()
    print(f"✅ Saved SMOTE effect plot to {FIGURES_DIR}/smote_effect.png")
    print(f"   SMOTE increased minority class from {np.bincount(y_train)[0]} to {np.bincount(y_train_res)[0]} samples.")

# ===========================================================================
# 4. MAIN
# ===========================================================================
def main():
    print("="*70)
    print(" COMPREHENSIVE EDA FOR DENGUE OUTBREAK DATA")
    print("="*70)

    # 1. Raw data analysis
    df_raw = analyze_raw_data()

    # 2. Merged data analysis
    df_merged, feature_cols = analyze_merged_data()

    # 3. SMOTE demonstration
    demonstrate_smote(df_merged, feature_cols)

    print("\n" + "="*70)
    print("🎉 ALL EDA COMPLETE. ALL PLOTS SAVED TO 'figures/' DIRECTORY.")
    print("="*70)

if __name__ == "__main__":
    main()