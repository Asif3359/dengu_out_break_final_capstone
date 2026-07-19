# Class Imbalance Handling – Complete Breakdown

Yes! **We handled class imbalance before every model train**, but the techniques were different for traditional ML vs. deep learning models.

---

## Class Imbalance Summary

| Class | Count | Percentage |
|-------|-------|------------|
| **Outbreak (1)** | 38,303 | **91.59%** |
| **No Outbreak (0)** | 3,517 | **8.41%** |

**Imbalance Ratio:** ~ **11:1** (severe imbalance)

---

## Handling Techniques – Model-Wise

### 1. Random Forest – `class_weight='balanced'`

```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',  # ← Handles imbalance internally
    random_state=42
)
```

**How it works:** Random Forest automatically assigns higher weight to minority class samples during training.

---

### 2. XGBoost – `sample_weight` (Class Weighting)

```python
from sklearn.utils.class_weight import compute_sample_weight

# Compute balanced weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    ...
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)  # ← Weighted training
```

**How it works:** Each sample gets a weight inversely proportional to its class frequency. Minority class samples get higher weight.

---

### 3. LightGBM – `scale_pos_weight`

```python
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos  # ≈ 0.09 (majority/minority)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,  # ← Handles imbalance natively
    ...
)
```

**How it works:** LightGBM internally uses `scale_pos_weight` to adjust the loss function.

---

### 4. Deep Learning (LSTM & Transformer) – SMOTE + Class Weights

```python
from imblearn.over_sampling import SMOTE

# Flatten sequences for SMOTE
n_samples, lookback, n_features = X_train.shape
X_train_flat = X_train.reshape(n_samples, lookback * n_features)

# Apply SMOTE – balance to 50:50
smote = SMOTE(random_state=42)
X_train_flat_res, y_train_res = smote.fit_resample(X_train_flat, y_train)

# Reshape back to sequences
X_train = X_train_flat_res.reshape(-1, lookback, n_features)

# Also apply class weights during training
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
class_weight_dict = dict(zip(np.unique(y_train_res), class_weights))

model.fit(X_train, y_train, class_weight=class_weight_dict)
```

**How it works:** 
- **SMOTE:** Creates synthetic minority samples (50:50 balance)
- **Class Weights:** Additional weighting during training

---

##  Summary Table – Imbalance Handling

| Model | Technique Used | Code Implementation |
|-------|---------------|---------------------|
| **Random Forest** | `class_weight='balanced'` | Built-in parameter |
| **XGBoost** | `sample_weight` | `compute_sample_weight()` + `sample_weight` parameter |
| **LightGBM** | `scale_pos_weight` | `scale_pos_weight = neg/pos` |
| **LSTM** | SMOTE + Class Weights | `SMOTE()` + `class_weight` parameter |
| **Transformer** | SMOTE + Class Weights | `SMOTE()` + `class_weight` parameter |

---

##  Why Different Techniques?

| Model Type | Why This Technique? |
|------------|---------------------|
| **Tree Models (RF, XGBoost, LGBM)** | Class weighting is simpler, faster, and doesn't create synthetic data. Tree models handle weights natively. |
| **Deep Learning (LSTM, Transformer)** | Neural networks are more sensitive to imbalance. SMOTE helps by creating additional training samples, preventing the model from ignoring the minority class. |

---

##  Threshold Optimization – For All Models

**After training, we also optimized the classification threshold** for every model:

```python
def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_macro_f1 = 0.0
    for thresh in thresholds:
        y_pred_tmp = (y_proba >= thresh).astype(int)
        f1_macro = f1_score(y_true, y_pred_tmp, average='macro')
        if f1_macro > best_macro_f1:
            best_macro_f1 = f1_macro
            best_threshold = thresh
    return best_threshold, best_macro_f1
```

**Results:**

| Model | Optimal Threshold | Default Threshold | Impact |
|-------|-------------------|-------------------|--------|
| **XGBoost** | **0.330** | 0.5 | Lowered to improve recall |
| Random Forest | 0.370 | 0.5 | Lowered to improve recall |
| LightGBM | 0.380 | 0.5 | Lowered to improve recall |
| LSTM | 0.460 | 0.5 | Slightly lowered |
| Transformer | 0.530 | 0.5 | Slightly raised |

---

##  Why XGBoost Was the Best Despite Imbalance

| Model | Macro F1 | Imbalance Handling | Why It Worked |
|-------|----------|-------------------|---------------|
| **XGBoost** | **0.695** | Sample weighting + threshold optimization | Best balance between classes |
| Random Forest | 0.681 | `class_weight='balanced'` | Close, but slightly underperformed |
| LightGBM | 0.566 | `scale_pos_weight` | Underperformed – possibly hyperparameter mismatch |
| Transformer | 0.560 | SMOTE + class weights | Underperformed – limited data |
| LSTM | 0.508 | SMOTE + class weights | Underperformed – limited data |

---

##  Quick Reference for Defense

**Question:** *"How did you handle class imbalance?"*

**Answer:**

> *"Our dataset had 91.6% positive class (outbreak) and only 8.4% negative class (no outbreak) – an 11:1 ratio. We used different techniques for different models:*
>
> - *For **Random Forest**, we used `class_weight='balanced'`.*
> - *For **XGBoost**, we used `sample_weight` from `compute_sample_weight('balanced')`.*
> - *For **LightGBM**, we used `scale_pos_weight = neg/pos`.*
> - *For **LSTM and Transformer**, we applied **SMOTE** to create synthetic minority samples (50:50 balance) and also used class weights.*
>
> *Additionally, we optimized the classification threshold for every model to maximize macro F1. The optimal threshold for XGBoost was 0.330 (vs default 0.5), which improved recall for the minority class."*

---

## Final Performance with Imbalance Handling

| Metric | Without Handling | With Handling (XGBoost) |
|--------|------------------|--------------------------|
| **Macro F1** | 0.45 (approx.) | **0.695** |
| **No Outbreak Recall** | ~0% | **40.8%** |
| **Outbreak Recall** | ~100% | **95.8%** |

**Key Takeaway:** Imbalance handling significantly improved the model's ability to detect non-outbreak weeks (from ~0% to 40.8%) while maintaining high outbreak recall (95.8%).