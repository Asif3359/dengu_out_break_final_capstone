#!/usr/bin/env python
# eda_analysis.py
"""
Comprehensive exploratory data analysis for the final merged dataset.
This script:
  1. Loads the dataset.
  2. Prints basic info, missing values (should be none).
  3. Analyzes target variable (dengue_total) and outbreak_flag.
  4. Generates statistical summaries.
  5. Plots distributions, correlations, feature importance.
  6. Saves all figures to the figures/ directory.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DATA_PATH = "data/merged_data.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
print("="*70)
print(" EXPLORATORY DATA ANALYSIS FOR DENGUE OUTBREAK DATASET")
print("="*70)

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"\n✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

# ===========================================================================
# 2. BASIC INFO & MISSING VALUES
# ===========================================================================
print("\n" + "-"*70)
print(" BASIC INFORMATION")
print("-"*70)
print(df.info())

print("\n🔍 Missing values per column:")
print(df.isnull().sum())

# ===========================================================================
# 3. TARGET VARIABLE ANALYSIS
# ===========================================================================
print("\n" + "-"*70)
print(" TARGET VARIABLE ANALYSIS")
print("-"*70)

# Create outbreak_flag (binary) for classification analysis
df['outbreak_flag'] = (df['dengue_total'] > 0).astype(int)

# Distribution of dengue_total
print("\n📊 Summary of 'dengue_total':")
print(df['dengue_total'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

# Class balance
outbreak_counts = df['outbreak_flag'].value_counts()
print("\n🦟 Outbreak vs Non-Outbreak counts:")
print(outbreak_counts)
print(f"   Percentage of outbreaks: {outbreak_counts[1] / len(df) * 100:.2f}%")

# Plot target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Histogram of dengue_total (log scale if needed)
ax1 = axes[0]
ax1.hist(df['dengue_total'], bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Dengue cases')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Dengue Cases')
ax1.grid(True, alpha=0.3)

# Boxplot by country (optional)
ax2 = axes[1]
sns.boxplot(data=df, x='ISO_A0', y='dengue_total', ax=ax2)
ax2.set_xlabel('Country')
ax2.set_ylabel('Dengue cases')
ax2.set_title('Dengue Cases by Country')
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/target_distribution.png", dpi=150)
plt.show()
print(f"✅ Saved target distribution plot to {FIGURES_DIR}/target_distribution.png")

# ===========================================================================
# 4. NUMERICAL FEATURE SUMMARY
# ===========================================================================
print("\n" + "-"*70)
print(" NUMERICAL FEATURE SUMMARY")
print("-"*70)

# Select only numeric columns (excluding non-numeric like ISO_A0)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove target and identifier columns that might not be features
exclude_cols = ['dengue_total', 'outbreak_flag', 'Year', 'time_index', 'year_normalized']
# Also exclude encoded country if you want but keep it
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"\n📈 Summary of {len(feature_cols)} numeric features (showing first 5):")
print(df[feature_cols].describe().T.head(10))

# Check for outliers using IQR
print("\n🔍 Outlier detection (IQR method) for top 5 features:")
for col in feature_cols[:5]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    print(f"   {col}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")

# ===========================================================================
# 5. CORRELATION MATRIX (for a subset of important features)
# ===========================================================================
print("\n" + "-"*70)
print(" FEATURE CORRELATION")
print("-"*70)

# Select a manageable subset of features (e.g., top 15) to avoid clutter
# We'll pick lags, weather, rolling stats, etc.
subset_cols = [
    'dengue_lag_1', 'dengue_lag_2', 'dengue_lag_3', 'dengue_lag_6', 'dengue_lag_12',
    'dengue_ma_3', 'dengue_ma_6', 'dengue_ma_12',
    'temperature_c', 'temperature_c_lag_1', 'temperature_c_lag_2', 'temperature_c_lag_3',
    'precipitation_mm', 'precipitation_mm_lag_1', 'precipitation_mm_lag_2', 'precipitation_mm_lag_3',
    'temp_precip_interaction',
    'month_sin', 'month_cos', 'is_rainy_season'
]
# Keep only those that exist in the dataframe
subset_cols = [col for col in subset_cols if col in df.columns]

# Compute correlation with target
corr_matrix = df[subset_cols + ['dengue_total']].corr()
# Sort by absolute correlation with dengue_total
target_corr = corr_matrix['dengue_total'].sort_values(ascending=False)
print("\n🔗 Top 10 features correlated with dengue_total (absolute value):")
print(target_corr.abs().sort_values(ascending=False).head(10))

# Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Key Features')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150)
plt.show()
print(f"✅ Saved correlation heatmap to {FIGURES_DIR}/correlation_heatmap.png")

# ===========================================================================
# 6. FEATURE IMPORTANCE USING RANDOM FOREST (Regression)
# ===========================================================================
print("\n" + "-"*70)
print(" FEATURE IMPORTANCE (RANDOM FOREST REGRESSOR)")
print("-"*70)

# Prepare X (features) and y (target)
X = df[feature_cols]  # all numeric features except target and id columns
y = df['dengue_total']

# Because of many zeros, we'll use a sample for speed if needed
# but we can train on full data (it's ~40k rows, manageable)
print(f"Training Random Forest on {len(X)} rows with {len(feature_cols)} features...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X, y)
importances = rf_reg.feature_importances_
feature_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

print("\n🏆 Top 10 most important features (regression):")
print(feature_imp.head(10))

# Plot
plt.figure(figsize=(10, 8))
feature_imp.head(15).plot(kind='barh', color='teal')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Random Forest Regressor)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/feature_importance_regression.png", dpi=150)
plt.show()
print(f"✅ Saved feature importance plot to {FIGURES_DIR}/feature_importance_regression.png")

# ===========================================================================
# 7. TIME SERIES PLOTS (by Country)
# ===========================================================================
print("\n" + "-"*70)
print(" TIME SERIES OVERVIEW")
print("-"*70)

df['calendar_start_date'] = pd.to_datetime(df['calendar_start_date'])
countries = df['ISO_A0'].unique()

fig, axes = plt.subplots(nrows=len(countries), ncols=1, figsize=(14, 3*len(countries)))
if len(countries) == 1:
    axes = [axes]
for ax, country in zip(axes, countries):
    sub = df[df['ISO_A0'] == country].sort_values('calendar_start_date')
    ax.plot(sub['calendar_start_date'], sub['dengue_total'], label=country, linewidth=1.5)
    ax.set_title(f'{country} - Dengue Cases Over Time')
    ax.set_ylabel('Cases')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/timeseries_by_country.png", dpi=150)
plt.show()
print(f"✅ Saved time series plots to {FIGURES_DIR}/timeseries_by_country.png")

# ===========================================================================
# 8. SEASONALITY PATTERNS (Average by month)
# ===========================================================================
print("\n" + "-"*70)
print(" SEASONALITY ANALYSIS")
print("-"*70)

# Group by month across all countries
monthly_avg = df.groupby('month')['dengue_total'].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(monthly_avg['month'], monthly_avg['dengue_total'], color='coral')
plt.xlabel('Month')
plt.ylabel('Average Dengue Cases')
plt.title('Average Dengue Cases by Month (all countries)')
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/seasonality_monthly.png", dpi=150)
plt.show()
print(f"✅ Saved seasonality plot to {FIGURES_DIR}/seasonality_monthly.png")

# ===========================================================================
# 9. CLASS IMBALANCE CHECK FOR OUTBREAK FLAG
# ===========================================================================
print("\n" + "-"*70)
print(" CLASS IMBALANCE CHECK")
print("-"*70)

# We already computed outbreak_counts. Also check per country.
print("\n📊 Outbreak distribution by country:")
country_outbreak = df.groupby('ISO_A0')['outbreak_flag'].mean().sort_values(ascending=False)
print(country_outbreak)

# Plot
plt.figure(figsize=(10, 5))
country_outbreak.plot(kind='bar', color='skyblue')
plt.xlabel('Country')
plt.ylabel('Proportion of Outbreaks')
plt.title('Outbreak Proportion by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/outbreak_proportion_by_country.png", dpi=150)
plt.show()
print(f"✅ Saved outbreak proportion plot to {FIGURES_DIR}/outbreak_proportion_by_country.png")

# ===========================================================================
# 10. CONCLUSION
# ===========================================================================
print("\n" + "="*70)
print(" EDA COMPLETE - SUMMARY")
print("="*70)
print(f"✅ Total rows: {len(df)}")
print(f"✅ Total features: {len(df.columns)}")
print(f"✅ Outbreak percentage: {outbreak_counts[1] / len(df) * 100:.2f}%")
print(f"✅ Top 3 important features (regression):")
print(feature_imp.head(3))
print("\n📁 All plots saved to 'figures/' directory.")
print("🎉 Ready to proceed with model training.")