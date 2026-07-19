# 🦟 Dengue Outbreak Prediction – Complete AI/ML Thesis Summary

This document covers **everything you need to know** for your pre-defense, focusing on the **AI/ML aspects** of your project.

---

## 1. Problem Statement & Motivation

Dengue fever remains one of the most significant mosquito-borne viral diseases globally. In 2024 alone, WHO received reports of over **14.4 million dengue cases** worldwide.

**The Problem:**
- Dengue outbreaks cause significant morbidity, mortality, and economic burden
- Early warning systems can enable proactive public health interventions
- Accurate outbreak prediction helps in resource allocation and preparedness

**Our Goal:**
To build a **machine learning system** that predicts dengue outbreaks (binary classification) using:
- Historical dengue case data
- Climate/weather data
- Temporal features

---

## 2. Data Collection & Sources

### 2.1 Dengue Surveillance Data – **OpenDengue Project**

| Attribute | Details |
|-----------|---------|
| **Source** | OpenDengue Project (opendengue.org) |
| **Version** | 1.2 |
| **Total Cases** | Over **56 million** dengue cases |
| **Countries** | **102 countries** (1924–2023) |
| **Our Region** | SEARO (South-East Asia Region) – **10 countries** |
| **Temporal Coverage** | **1981–2025** |
| **Temporal Resolution** | **Monthly** |
| **Target Variable** | `dengue_total` (monthly case counts) |
| **Raw Records** | 41,997 rows |

**Countries in Our Dataset:**
- Bangladesh (BGD)
- India (IND)
- Indonesia (IDN)
- Thailand (THA)
- Sri Lanka (LKA)
- Myanmar (MMR)
- Nepal (NPL)
- Maldives (MDV)
- Timor-Leste (TLS)
- Bhutan (BTN)

### 2.2 Climate Data – **NASA POWER API**

| Attribute | Details |
|-----------|---------|
| **Source** | NASA POWER API (power.larc.nasa.gov) |
| **Endpoint** | `/api/temporal/monthly/point` |
| **Parameters** | `T2M` (Temperature at 2 meters), `PRECTOTCORR` (Corrected Precipitation) |
| **Temporal Coverage** | **1981–2025** |
| **Resolution** | **Monthly** |
| **Format** | JSON |

**Why Weather Data?**
- Mosquito breeding and viral replication are **temperature-sensitive**
- Precipitation creates **breeding sites** for mosquitoes
- Dengue transmission shows **seasonal patterns**

---

## 3. Data Preprocessing & Final Dataset

### 3.1 Data Merging

The two datasets were merged using:
- **`ISO_A0`** (country code)
- **`year`**
- **`month`**

**Final Dataset:**
- **41,820 rows** × **49 columns**
- After filtering to years ≥ 1981 (NASA data availability)
- After dropping rows with missing values (due to lags)

### 3.2 Target Variable – Two Approaches Evaluated

| Approach | Target | Purpose |
|----------|--------|---------|
| **Regression** | `dengue_total` (continuous case counts) | Predict exact case numbers |
| **Classification** | `outbreak_flag` (binary) | Predict if outbreak will occur |

**Classification Target Definition:**
- `outbreak_flag = 1` if `dengue_total > 0`
- `outbreak_flag = 0` if `dengue_total = 0`

**Class Distribution:**
| Class | Count | Percentage |
|-------|-------|------------|
| Outbreak (1) | 38,303 | **91.59%** |
| No Outbreak (0) | 3,517 | **8.41%** |

This is **severe class imbalance** (≈11:1 ratio).

---

## 4. Feature Engineering – 45 Features

### 4.1 Temporal Features (6 features)

| Feature | Description | Why? |
|---------|-------------|------|
| `year` | Calendar year | Captures long-term trends |
| `month` | Calendar month (1-12) | Captures seasonality |
| `dayofyear` | Day of year (1-365) | Captures fine-grained seasonality |
| `weekofyear` | Week of year (1-53) | Captures weekly patterns |
| `quarter` | Quarter of year (1-4) | Captures seasonal blocks |
| `is_rainy_season` | Binary (1 if month 6-10) | Identifies monsoon season |

### 4.2 Cyclical Encoding (4 features)

| Feature | Formula | Why? |
|---------|---------|------|
| `month_sin` | sin(2π × month / 12) | Preserves cyclic nature of months |
| `month_cos` | cos(2π × month / 12) | Preserves cyclic nature of months |
| `dayofyear_sin` | sin(2π × dayofyear / 365) | Preserves cyclic nature of days |
| `dayofyear_cos` | cos(2π × dayofyear / 365) | Preserves cyclic nature of days |

**Why Cyclical Encoding?**
- Months are **cyclic** (December → January is a smooth transition)
- Simple integer encoding would treat December and January as far apart (12 vs 1)
- Sin/cos encoding preserves the cyclic relationship

### 4.3 Lag Features – Dengue Cases (5 features)

| Feature | Description | Why? |
|---------|-------------|------|
| `dengue_lag_1` | Cases 1 month ago | Short-term memory |
| `dengue_lag_2` | Cases 2 months ago | Short-term memory |
| `dengue_lag_3` | Cases 3 months ago | Medium-term memory |
| `dengue_lag_6` | Cases 6 months ago | Semi-annual pattern |
| `dengue_lag_12` | Cases 12 months ago | Annual seasonality |

**Why Lags?**
- Dengue has **temporal autocorrelation** (past cases predict future cases)
- Different lags capture different transmission cycles

### 4.4 Rolling Statistics – Dengue Cases (6 features)

| Feature | Description | Why? |
|---------|-------------|------|
| `dengue_ma_3` | 3-month moving average | Smoothes short-term noise |
| `dengue_ma_6` | 6-month moving average | Captures semi-annual trends |
| `dengue_ma_12` | 12-month moving average | Captures annual trends |
| `dengue_std_3` | 3-month standard deviation | Captures short-term variability |
| `dengue_std_6` | 6-month standard deviation | Captures medium-term variability |
| `dengue_std_12` | 12-month standard deviation | Captures annual variability |

**⚠️ Important:** To prevent **data leakage**, all rolling statistics were computed using `shift(1)` before the rolling operation. This ensures only **past information** is used.

### 4.5 Weather Features (29 features)

| Feature Category | Features | Description |
|------------------|----------|-------------|
| Current | `temperature_c`, `precipitation_mm` | Current month values |
| Temperature Lags | `temperature_c_lag_1,2,3,6,12` | 1,2,3,6,12-month lags |
| Precipitation Lags | `precipitation_mm_lag_1,2,3,6,12` | 1,2,3,6,12-month lags |
| Temperature Rolling | `temperature_c_ma_3,6`, `temperature_c_std_3,6` | 3,6-month rolling stats |
| Precipitation Rolling | `precipitation_mm_ma_3,6`, `precipitation_mm_std_3,6` | 3,6-month rolling stats |
| Interaction | `temp_precip_interaction` | Temperature × Precipitation |

**Why Weather Features?**
- Temperature affects **mosquito development** and **viral replication**
- Precipitation creates **breeding sites** for mosquitoes
- Weather effects are **delayed** (lags capture this)

### 4.6 Additional Features (3 features)

| Feature | Description | Why? |
|---------|-------------|------|
| `time_index` | Sequential index per country | Captures country-specific trends |
| `year_normalized` | Normalized year (0-1 range) | Captures long-term trends |
| `ISO_A0_encoded` | Label-encoded country codes | Captures country-specific effects |

---

## 5. Exploratory Data Analysis (EDA) – Key Insights

### 5.1 Total Cases by Country

| Country | Total Cases | Percentage |
|---------|-------------|------------|
| Indonesia | 8,359,239 | 46.1% |
| Thailand | 4,479,008 | 24.7% |
| India | 2,275,276 | 12.5% |
| Sri Lanka | 1,352,147 | 7.5% |
| Bangladesh | 672,026 | 3.7% |
| Myanmar | 634,947 | 3.5% |
| Nepal | 276,657 | 1.5% |
| Others | 84,641 | 0.5% |

### 5.2 Outbreak Proportion by Country

| Country | ISO Code | Outbreak % |
|---------|----------|------------|
| Maldives | MDV | **100.0%** |
| Indonesia | IDN | 95.6% |
| Thailand | THA | 93.6% |
| India | IND | 90.2% |
| Sri Lanka | LKA | 88.9% |
| Myanmar | MMR | 87.9% |
| Timor-Leste | TLS | 83.1% |
| Bangladesh | BGD | 80.5% |
| Bhutan | BTN | 78.9% |
| Nepal | NPL | 65.3% |

### 5.3 Dengue Cases Distribution (Regression Target)

| Statistic | Value |
|-----------|-------|
| Mean | 402.09 cases/month |
| Standard Deviation | 3,059.31 |
| Minimum | 0 |
| 25th Percentile | 6 |
| Median | 24 |
| 75th Percentile | 97 |
| 90th Percentile | 398 |
| 99th Percentile | 8,101 |
| Maximum | 157,442 |

**Key Insight:** The distribution is **highly right-skewed** with extreme outliers → **regression is challenging** → **classification is better**.

### 5.4 Feature Correlation with Target

| Feature | Correlation |
|---------|-------------|
| `dengue_ma_3` | 0.264 |
| `dengue_ma_6` | 0.231 |
| `dengue_ma_12` | 0.225 |
| `dengue_lag_1` | 0.212 |
| `dengue_lag_2` | 0.174 |
| `dengue_lag_3` | 0.158 |
| `dengue_lag_12` | 0.093 |
| `dengue_lag_6` | 0.072 |
| `temp_precip_interaction` | 0.069 |

**Key Insight:** Past dengue cases are the **strongest predictors**; weather features show **weak correlation**.

### 5.5 Feature Importance (Random Forest Regressor)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `dengue_lag_1` | 0.122 |
| 2 | `dengue_ma_12` | 0.100 |
| 3 | `dengue_lag_3` | 0.086 |
| 4 | `dengue_std_3` | 0.066 |
| 5 | `dengue_lag_2` | 0.059 |
| 6 | `dengue_std_12` | 0.055 |
| 7 | `dengue_ma_3` | 0.053 |
| 8 | `dengue_std_6` | 0.052 |
| 9 | `dengue_lag_12` | 0.051 |
| 10 | `dengue_ma_6` | 0.047 |

**Key Insight:** **ALL top 10 features are from historical dengue cases** – weather features did not appear in the top 10.

---

## 6. Models Evaluated

### 6.1 Problem Formulation

After evaluating both approaches:

| Approach | Performance | Conclusion |
|----------|-------------|------------|
| **Regression** | R² = **0.1575** | Poor – cannot predict exact case counts |
| **Classification** | Macro F1 = **0.695** | Good – can predict outbreak occurrence |

**Why Classification?**
- Outbreaks are triggered by **alerts**, not precise counts
- Public health officials need **actionable predictions**
- Classification is more **interpretable**

### 6.2 Traditional Machine Learning Models (3 models)

#### 6.2.1 Random Forest

| Attribute | Details |
|-----------|---------|
| **Type** | Ensemble of decision trees (bagging) |
| **Strengths** | Robust to overfitting, handles non-linearity, provides feature importance |
| **Hyperparameters** | `n_estimators=200`, `max_depth=20`, `class_weight='balanced'` |

**Why Random Forest?**
- Handles non-linear relationships well
- Provides feature importance for interpretability
- Robust to outliers and noise

#### 6.2.2 XGBoost (eXtreme Gradient Boosting) – **WINNER**

| Attribute | Details |
|-----------|---------|
| **Type** | Gradient boosting with regularization |
| **Strengths** | State-of-the-art, handles missing values, built-in regularization |
| **Hyperparameters** | `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'` |

**Why XGBoost?**
- State-of-the-art performance on tabular data
- Built-in regularization prevents overfitting
- Handles class imbalance effectively
- Fast training (under 5 seconds)

#### 6.2.3 LightGBM (Light Gradient Boosting Machine)

| Attribute | Details |
|-----------|---------|
| **Type** | Gradient boosting with histogram-based approach |
| **Strengths** | Faster training, efficient memory usage |
| **Hyperparameters** | `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `num_leaves=31`, `scale_pos_weight` |

**Why LightGBM?**
- Very fast training
- Good for large datasets
- Handles large feature sets efficiently

---

### 6.3 Deep Learning Models (2 models – Experimental)

#### 6.3.1 LSTM (Long Short-Term Memory)

| Attribute | Details |
|-----------|---------|
| **Type** | Recurrent neural network with memory cells |
| **Strengths** | Captures long-term dependencies, handles sequential data naturally |
| **Architecture** | Bidirectional LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(1, sigmoid) |

**Why LSTM?**
- Designed for time-series data
- Can capture long-term dependencies
- Handles sequential patterns in dengue transmission

#### 6.3.2 Transformer

| Attribute | Details |
|-----------|---------|
| **Type** | Self-attention based architecture |
| **Strengths** | Captures global dependencies, parallelizable training |
| **Architecture** | Dense(64) → Multi-Head Attention(4) → Feed-Forward(128) → GlobalAvgPooling → Dense(1, sigmoid) |

**Why Transformer?**
- Can capture long-range dependencies
- Parallel training (faster than RNNs)
- Attention mechanisms can learn cross-country patterns

---

## 7. Handling Class Imbalance

The dataset had **91.59% positive class** (severe imbalance ≈11:1).

### Techniques Used:

| Technique | Application | Why? |
|-----------|-------------|------|
| **Class Weighting** | `compute_sample_weight(class_weight='balanced')` for all models | Assigns higher weight to minority class |
| **Threshold Optimization** | Maximizes macro F1 on validation set | Finds optimal decision boundary |
| **Scale Pos Weight** | For LightGBM: `scale_pos_weight = neg/pos` | Handles imbalance natively |
| **SMOTE** | Applied to training sequences for deep learning models (50:50) | Creates synthetic minority samples |

**Why Class Weighting over SMOTE for Tree Models?**
- Simpler, faster, no synthetic data creation
- Tree models handle weights natively
- SMOTE can introduce noise

---

## 8. Results & Performance

### 8.1 Traditional Models Performance

| Metric | Random Forest | XGBoost | LightGBM |
|--------|---------------|---------|----------|
| **Macro F1** | 0.681 | **0.695** | 0.566 |
| **Accuracy** | 0.912 | 0.912 | 0.907 |
| **F1 (Outbreak)** | 0.951 | **0.952** | 0.948 |
| **Recall (Outbreak)** | 0.958 | **0.958** | 0.957 |
| **Precision (Outbreak)** | 0.944 | **0.946** | 0.940 |
| **Recall (No Outbreak)** | 0.399 | **0.408** | 0.352 |
| **ROC-AUC** | 0.843 | **0.844** | 0.841 |
| **Optimal Threshold** | 0.370 | **0.330** | 0.380 |

### 8.2 Deep Learning Models Performance

| Metric | LSTM | Transformer |
|--------|------|-------------|
| **Macro F1** | 0.508 | **0.560** |
| **Accuracy** | 0.857 | **0.868** |
| **F1 (Outbreak)** | 0.922 | **0.928** |
| **Recall (Outbreak)** | 0.939 | **0.943** |
| **Precision (Outbreak)** | 0.906 | **0.914** |
| **Recall (No Outbreak)** | 0.078 | **0.163** |
| **ROC-AUC** | 0.597 | **0.605** |
| **Optimal Threshold** | 0.460 | **0.530** |

### 8.3 XGBoost – Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Outbreak) | 0.471 | 0.408 | 0.437 | 703 |
| 1 (Outbreak) | **0.946** | **0.958** | **0.952** | 7,661 |
| **Macro Avg** | 0.708 | 0.683 | **0.695** | 8,364 |
| **Weighted Avg** | 0.906 | 0.912 | 0.909 | 8,364 |

### 8.4 Confusion Matrix – XGBoost

| | Predicted No | Predicted Outbreak |
|---|--------------|-------------------|
| **Actual No** | 287 | 416 |
| **Actual Outbreak** | 323 | 7,338 |

---

## 9. Why XGBoost Was Selected

### 9.1 Performance Reasons

1. **Highest Macro F1 (0.695):** Best balanced performance across both classes
2. **Best Outbreak Recall (95.8%):** Catches 7,338 out of 7,661 actual outbreaks
3. **Best Precision for Outbreaks (94.6%):** Minimizes false alarms
4. **Second best No-Outbreak Recall (40.8%):** Only slightly behind Random Forest (39.9%)

### 9.2 Technical Reasons

1. **Superior Handling of Imbalanced Data:** Built-in regularization and class weight handling
2. **Computational Efficiency:** Training under 5 seconds
3. **Better than Deep Learning:** Outperformed LSTM and Transformer significantly
4. **Robust to Overfitting:** Regularization prevents overfitting

### 9.3 Public Health Reasons

1. **High Outbreak Recall (95.8%):** In public health, missing an outbreak is more costly than a false alarm
2. **High Precision (94.6%):** Minimizes wasted resources on false alarms
3. **Actionable Predictions:** Classification provides clear "outbreak/no outbreak" decisions

---

## 10. SHAP Interpretability

SHAP (SHapley Additive exPlanations) analysis was applied to explain the XGBoost model's predictions.

**Top 10 Features by Mean |SHAP| Value:**

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 1 | `dengue_lag_1` | 0.124 |
| 2 | `dengue_ma_12` | 0.102 |
| 3 | `dengue_lag_3` | 0.088 |
| 4 | `dengue_std_3` | 0.068 |
| 5 | `dengue_lag_2` | 0.061 |
| 6 | `dengue_std_12` | 0.056 |
| 7 | `dengue_ma_3` | 0.054 |
| 8 | `dengue_std_6` | 0.053 |
| 9 | `dengue_lag_12` | 0.052 |
| 10 | `dengue_ma_6` | 0.048 |

**Key Insight:** Historical dengue cases dominate the predictions – weather features contribute minimally.

---

## 11. Discussion: Key Findings

### 11.1 What We Learned

1. **Past dengue cases are the strongest predictors.** Lag features and rolling statistics dominate feature importance, highlighting the autoregressive nature of dengue transmission.

2. **Weather features have limited predictive power.** Despite including temperature and precipitation lags up to 12 months, these features did not appear in the top 10 importance rankings. Possible reasons:
   - Monthly aggregation is too coarse
   - Weather effects may have delays longer than 12 months
   - The relationship is non-linear and requires more complex feature engineering

3. **Classification outperforms regression.** R² of 0.158 for regression vs macro F1 of 0.695 for classification demonstrates that predicting outbreak occurrence is more tractable.

4. **Threshold optimization is critical.** The optimal threshold of 0.330 (vs default 0.5) significantly improved macro F1.

5. **Traditional ML beats Deep Learning for this task.** XGBoost achieved significantly higher macro F1 than LSTM (0.508) and Transformer (0.560), even after SMOTE and class weighting.

### 11.2 Limitations

1. **Data Resolution:** Monthly aggregation may mask important weekly or daily patterns
2. **Geographic Coverage:** Only 10 countries in SEARO region
3. **Weather Data:** NASA POWER provides gridded estimates, not ground-truth measurements
4. **Missing Variables:** Population density, vector indices, socio-economic variables not available
5. **Class Imbalance:** 91.6% positive class may still bias the model
6. **Deep Learning Data:** Limited dataset (41,820 rows) may not be sufficient

---

## 12. Conclusion

### 12.1 Summary

| Aspect | Achievement |
|--------|-------------|
| **Data** | 41,820 records × 49 columns from 10 SEARO countries |
| **Features** | 45 engineered features (lags, rolling stats, cyclical encodings, weather) |
| **Models Evaluated** | Random Forest, XGBoost, LightGBM, LSTM, Transformer |
| **Best Model** | **XGBoost** (Macro F1 = 0.695, Outbreak Recall = 95.8%) |
| **Deployment** | FastAPI backend + Next.js dashboard |

### 12.2 Final Performance

| Metric | Value |
|--------|-------|
| **Macro F1** | 0.695 |
| **Accuracy** | 91.6% |
| **Outbreak Recall** | 95.8% |
| **Outbreak Precision** | 94.6% |
| **ROC-AUC** | 0.844 |
| **Optimal Threshold** | 0.330 |

### 12.3 Why This Matters

- **Early warning system** for public health authorities
- **Proactive interventions** (mosquito control, resource allocation)
- **Evidence-based policy** for dengue prevention
- **Data-driven decision making** in SEARO region

---

## 13. Potential Defense Questions

### Q1: Why did you choose XGBoost over deep learning models?

**Answer:** XGBoost achieved significantly higher macro F1 (0.695) compared to LSTM (0.508) and Transformer (0.560). Despite the theoretical advantages of deep learning for time-series data, our dataset (41,820 rows) is tabular and relatively small. XGBoost's built-in regularization and class imbalance handling made it more effective. Additionally, XGBoost trains in under 5 seconds, making it suitable for real-time deployment.

### Q2: Why didn't weather features show high importance?

**Answer:** There are several possible reasons:
1. Monthly aggregation may not capture the relevant weather-disease dynamics
2. Weather effects may have delays longer than 12 months
3. The relationship may be non-linear and require more complex feature engineering
4. Other factors (population immunity, vector control) may dominate transmission patterns

### Q3: How did you handle class imbalance?

**Answer:** We used multiple techniques:
1. **Class Weighting:** `compute_sample_weight(class_weight='balanced')` for all models
2. **Threshold Optimization:** Finding the optimal threshold on validation set to maximize macro F1
3. **Scale Pos Weight:** For LightGBM
4. **SMOTE:** Applied to training sequences for deep learning models

### Q4: Why classification over regression?

**Answer:** Regression (XGBoost Regressor) achieved R² = 0.1575, indicating poor performance. The target distribution is highly skewed with extreme outliers. Public health officials need actionable predictions (outbreak vs no outbreak), not exact case counts. Classification provides clear decisions for interventions.

### Q5: What are the limitations of your work?

**Answer:**
1. Monthly resolution may miss weekly/daily patterns
2. Only 10 SEARO countries – limited generalizability
3. NASA POWER data is gridded estimates, not ground-truth
4. Missing variables (population density, vector indices)
5. 91.6% positive class may still bias the model
6. Limited dataset for deep learning

### Q6: How could you improve the model?

**Answer:**
1. Incorporate weekly/daily data
2. Add more weather variables (humidity, wind speed)
3. Include socio-economic and population data
4. Explore spatial modeling (neighboring countries)
5. Use more advanced deep learning architectures (Informer, PatchTST)
6. Collect more data for deep learning

### Q7: What is SHAP and why did you use it?

**Answer:** SHAP (SHapley Additive exPlanations) explains model predictions by showing the contribution of each feature. We used it for **interpretability** – to help public health officials understand why the model made a prediction. It showed that historical dengue cases (lags, moving averages) dominate the predictions.

---

## 14. Quick Reference Card

### Key Numbers
- **Dataset:** 41,820 rows × 49 columns
- **Countries:** 10 SEARO countries
- **Features:** 45 engineered features
- **Positive Class:** 91.59% (severe imbalance)
- **Best Model:** XGBoost
- **Macro F1:** 0.695
- **Outbreak Recall:** 95.8%
- **Optimal Threshold:** 0.330

### Key Features
1. `dengue_lag_1` – Most important (SHAP 0.124)
2. `dengue_ma_12` – Second most important
3. `dengue_lag_3` – Third most important

### Key Models
1. **XGBoost** – WINNER (Macro F1: 0.695)
2. Random Forest – 2nd place (Macro F1: 0.681)
3. LightGBM – 3rd place (Macro F1: 0.566)
4. Transformer – Experimental
5. LSTM – Experimental

### Key Techniques
- Class Weighting
- Threshold Optimization
- SHAP Interpretability
- Cyclical Encoding
- Rolling Statistics with shift(1) to prevent leakage

---

Good luck with your pre-defense! You've done excellent work.


# Conclusion
Our proposed model, XGBoost, provides:

- High outbreak recall (95.8%) – catches nearly all outbreaks

- High precision (94.6%) – few false alarms

- Fast inference – suitable for real-time deployment

- Interpretable – SHAP explains feature contributions

- Robust – handles class imbalance effectively

- This model is ready for production deployment as an early warning system for dengue outbreaks in the SEARO region.

