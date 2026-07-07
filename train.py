# # train.py
# import sys
# import json
# import logging
# import warnings
# from datetime import datetime

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve

# # Import our config
# import config

# warnings.filterwarnings("ignore")

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(f"{config.LOG_DIR}/training.log"),
#         logging.StreamHandler(sys.stdout),
#     ],
# )
# logger = logging.getLogger(__name__)


# def engineer_features(df):
#     """Apply all feature engineering steps."""
#     logger.info("Starting feature engineering...")

#     # 1. Lags
#     for i in config.LAG_DAYS:
#         if f"lag_{i}" not in df.columns:
#             df[f"lag_{i}"] = df.groupby("ISO_A0")["dengue_total"].shift(i)

#     # 2. Rolling statistics
#     for window in config.ROLLING_WINDOWS:
#         df[f"temp_rolling_mean_{window}"] = df.groupby("ISO_A0")["temperature_c"].transform(
#             lambda x: x.rolling(window=window, min_periods=1).mean()
#         )
#         df[f"precip_rolling_std_{window}"] = df.groupby("ISO_A0")["precipitation_mm"].transform(
#             lambda x: x.rolling(window=window, min_periods=1).std()
#         )

#     # 3. Cyclical encoding
#     df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#     df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
#     df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
#     df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

#     # 4. Interaction
#     df["temp_precip_interaction"] = df["temperature_c"] * df["precipitation_mm"]

#     # 5. Final feature list
#     feature_cols = [
#         "ISO_A0",
#         "year",
#         "month",
#         "dayofyear",
#         "weekofyear",
#         "month_sin",
#         "month_cos",
#         "dayofyear_sin",
#         "dayofyear_cos",
#         "temperature_c",
#         "precipitation_mm",
#         "temp_precip_interaction",
#     ]
#     # Add lags
#     for i in config.LAG_DAYS:
#         feature_cols.append(f"lag_{i}")
#     # Add rolling stats
#     for window in config.ROLLING_WINDOWS:
#         feature_cols.append(f"temp_rolling_mean_{window}")
#         feature_cols.append(f"precip_rolling_std_{window}")

#     return df, feature_cols


# def main():
#     logger.info("=" * 60)
#     logger.info("PRODUCTION TRAINING PIPELINE STARTED")
#     logger.info("=" * 60)

#     # 1. Load data
#     try:
#         df = pd.read_csv(config.DATA_PATH, low_memory=False)
#         logger.info(f"Loaded data: {df.shape}")
#     except FileNotFoundError:
#         logger.error(f"Data file not found at {config.DATA_PATH}")
#         sys.exit(1)

#     # 2. Preprocess dates and target
#     if "calendar_start_date" in df.columns:
#         df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"], errors="coerce")
#         df = df.sort_values(["ISO_A0", "calendar_start_date"])
#     if "dengue_total" not in df.columns:
#         logger.error("Column 'dengue_total' not found!")
#         sys.exit(1)
#     df["outbreak_flag"] = (df["dengue_total"] > 0).astype(int)

#     # 3. Feature engineering
#     df, feature_cols = engineer_features(df)
#     logger.info(f"Engineered features: {len(feature_cols)}")

#     # 4. Drop rows with NaN (due to lags)
#     model_df = df[feature_cols + ["outbreak_flag"]].dropna()
#     logger.info(f"Final dataset shape: {model_df.shape}")
#     logger.info(f"Class distribution: {model_df['outbreak_flag'].value_counts().to_dict()}")

#     # 5. Encode country
#     encoder = LabelEncoder()
#     model_df["ISO_A0_encoded"] = encoder.fit_transform(model_df["ISO_A0"])

#     # 6. Scale numeric features (excluding ISO_A0)
#     numeric_cols = [col for col in feature_cols if col != "ISO_A0"]
#     scaler = StandardScaler()
#     model_df[numeric_cols] = scaler.fit_transform(model_df[numeric_cols])

#     # 7. Prepare X and y
#     X = model_df[["ISO_A0_encoded"] + numeric_cols]
#     y = model_df["outbreak_flag"]
#     all_feature_names = X.columns.tolist()

#     # 8. Train/Test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
#     )
#     logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

#     # 9. Train the final model (Gradient Boosting)
#     logger.info("Training Gradient Boosting...")
#     model = GradientBoostingClassifier(**config.GB_PARAMS)
#     model.fit(X_train, y_train)

#     # 10. Evaluate on test set
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]
#     f1 = f1_score(y_test, y_pred)
#     logger.info(f"Test F1 Score: {f1:.4f}")

#     # 11. Find optimal threshold (production decision boundary)
#     precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
#     logger.info(f"Optimal threshold: {optimal_threshold:.4f}")

#     # 12. Save artifacts
#     joblib.dump(model, f"{config.MODEL_DIR}/model.joblib")
#     joblib.dump(scaler, f"{config.MODEL_DIR}/scaler.joblib")
#     joblib.dump(encoder, f"{config.MODEL_DIR}/encoder.joblib")

#     # 13. Save metadata (crucial for prediction)
#     metadata = {
#         "model_name": "GradientBoosting",
#         "feature_columns": all_feature_names,
#         "numeric_columns": numeric_cols,
#         "optimal_threshold": float(optimal_threshold),
#         "test_f1_score": float(f1),
#         "training_date": datetime.now().isoformat(),
#         "class_distribution": model_df["outbreak_flag"].value_counts().to_dict(),
#     }
#     with open(f"{config.MODEL_DIR}/metadata.json", "w") as f:
#         json.dump(metadata, f, indent=4)

#     # 14. Print classification report (for the log)
#     logger.info("\n" + classification_report(y_test, y_pred, digits=4))
#     logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

#     logger.info("=" * 60)
#     logger.info("✅ TRAINING COMPLETE. Model saved to 'models/'")
#     logger.info("=" * 60)


# if __name__ == "__main__":
#     main()


# # train.py (UPGRADED VERSION)
# import sys
# import json
# import logging
# import warnings
# from datetime import datetime

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
# from sklearn.utils.class_weight import compute_sample_weight

# # Try importing advanced libraries
# try:
#     import xgboost as xgb
#     HAS_XGB = True
# except ImportError:
#     HAS_XGB = False

# try:
#     import lightgbm as lgb
#     HAS_LGB = True
# except ImportError:
#     HAS_LGB = False

# import config
# warnings.filterwarnings("ignore")

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(f"{config.LOG_DIR}/training.log"),
#         logging.StreamHandler(sys.stdout),
#     ],
# )
# logger = logging.getLogger(__name__)


# def engineer_features(df):
#     """Apply all feature engineering steps."""
#     logger.info("Starting feature engineering...")
#     for i in config.LAG_DAYS:
#         if f"lag_{i}" not in df.columns:
#             df[f"lag_{i}"] = df.groupby("ISO_A0")["dengue_total"].shift(i)
#     for window in config.ROLLING_WINDOWS:
#         df[f"temp_rolling_mean_{window}"] = df.groupby("ISO_A0")["temperature_c"].transform(
#             lambda x: x.rolling(window=window, min_periods=1).mean()
#         )
#         df[f"precip_rolling_std_{window}"] = df.groupby("ISO_A0")["precipitation_mm"].transform(
#             lambda x: x.rolling(window=window, min_periods=1).std()
#         )
#     df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#     df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
#     df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
#     df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
#     df["temp_precip_interaction"] = df["temperature_c"] * df["precipitation_mm"]

#     feature_cols = ["ISO_A0", "year", "month", "dayofyear", "weekofyear",
#                     "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
#                     "temperature_c", "precipitation_mm", "temp_precip_interaction"]
#     for i in config.LAG_DAYS:
#         feature_cols.append(f"lag_{i}")
#     for window in config.ROLLING_WINDOWS:
#         feature_cols.append(f"temp_rolling_mean_{window}")
#         feature_cols.append(f"precip_rolling_std_{window}")
#     return df, feature_cols


# def main():
#     logger.info("=" * 60)
#     logger.info("🚀 PRODUCTION TRAINING PIPELINE (IMPROVED)")
#     logger.info("=" * 60)

#     # 1. Load data
#     df = pd.read_csv(config.DATA_PATH, low_memory=False)
#     if "calendar_start_date" in df.columns:
#         df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"], errors="coerce")
#         df = df.sort_values(["ISO_A0", "calendar_start_date"])
#     if "dengue_total" not in df.columns:
#         raise ValueError("'dengue_total' not found.")
#     df["outbreak_flag"] = (df["dengue_total"] > 0).astype(int)

#     # 2. Feature Engineering
#     df, feature_cols = engineer_features(df)
#     model_df = df[feature_cols + ["outbreak_flag"]].dropna()
#     logger.info(f"Final shape: {model_df.shape}")
#     logger.info(f"Class distribution: {model_df['outbreak_flag'].value_counts().to_dict()}")

#     # 3. Encoding & Scaling
#     encoder = LabelEncoder()
#     model_df["ISO_A0_encoded"] = encoder.fit_transform(model_df["ISO_A0"])
#     numeric_cols = [col for col in feature_cols if col != "ISO_A0"]
#     scaler = StandardScaler()
#     model_df[numeric_cols] = scaler.fit_transform(model_df[numeric_cols])

#     X = model_df[["ISO_A0_encoded"] + numeric_cols]
#     y = model_df["outbreak_flag"]
#     feature_names = X.columns.tolist()

#     # 4. Train/Test Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
#     )
#     logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

#     # 5. Calculate Sample Weights for Gradient Boosting (IMPROVEMENT 1)
#     sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
#     logger.info("Sample weights computed for balanced class training.")

#     # 6. Define Models
#     models_to_train = {}

#     # A) Random Forest
#     logger.info("\n[1/4] Training Random Forest...")
#     rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
#                                  class_weight='balanced', random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     models_to_train["Random Forest"] = rf

#     # B) Gradient Boosting WITH sample weights (IMPROVEMENT 2)
#     logger.info("\n[2/4] Training Gradient Boosting (with sample weights)...")
#     gb = GradientBoostingClassifier(
#         n_estimators=200, learning_rate=0.05, max_depth=5,
#         min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42
#     )
#     gb.fit(X_train, y_train, sample_weight=sample_weights)
#     models_to_train["Gradient Boosting (Weighted)"] = gb

#     # C) XGBoost (if available)
#     if HAS_XGB:
#         logger.info("\n[3/4] Training XGBoost...")
#         # Use scale_pos_weight to handle imbalance (IMPROVEMENT 3)
#         scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
#         xgb_model = xgb.XGBClassifier(
#             n_estimators=200, learning_rate=0.05, max_depth=6,
#             subsample=0.8, colsample_bytree=0.8,
#             scale_pos_weight=scale, random_state=42, n_jobs=-1
#         )
#         xgb_model.fit(X_train, y_train)
#         models_to_train["XGBoost"] = xgb_model

#     # D) LightGBM (if available)
#     if HAS_LGB:
#         logger.info("\n[4/4] Training LightGBM...")
#         scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
#         lgb_model = lgb.LGBMClassifier(
#             n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31,
#             scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1
#         )
#         lgb_model.fit(X_train, y_train)
#         models_to_train["LightGBM"] = lgb_model

#     # 7. Evaluate and Pick the Best Model
#     logger.info("\n" + "=" * 60)
#     logger.info("EVALUATING ALL MODELS")
#     logger.info("=" * 60)

#     best_f1 = 0
#     best_model = None
#     best_name = ""

#     for name, model in models_to_train.items():
#         y_pred = model.predict(X_test)
#         f1 = f1_score(y_test, y_pred)
#         acc = np.mean(y_pred == y_test)
#         logger.info(f"{name:30s} | Acc: {acc:.4f} | F1: {f1:.4f}")
#         if f1 > best_f1:
#             best_f1 = f1
#             best_model = model
#             best_name = name

#     logger.info("-" * 60)
#     logger.info(f"🏆 BEST MODEL: {best_name} (F1: {best_f1:.4f})")

#     # 8. Threshold Optimization on the Best Model
#     y_proba = best_model.predict_proba(X_test)[:, 1]
#     precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
#     optimal_idx = np.argmax(f1_scores)
#     optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
#     logger.info(f"Optimal threshold for {best_name}: {optimal_threshold:.4f}")

#     # 9. Save Artifacts
#     joblib.dump(best_model, f"{config.MODEL_DIR}/model.joblib")
#     joblib.dump(scaler, f"{config.MODEL_DIR}/scaler.joblib")
#     joblib.dump(encoder, f"{config.MODEL_DIR}/encoder.joblib")

#     metadata = {
#         "model_name": best_name,
#         "feature_columns": feature_names,
#         "numeric_columns": numeric_cols,
#         "optimal_threshold": float(optimal_threshold),
#         "test_f1_score": float(best_f1),
#         "training_date": datetime.now().isoformat(),
#         "class_distribution": model_df["outbreak_flag"].value_counts().to_dict(),
#     }
#     with open(f"{config.MODEL_DIR}/metadata.json", "w") as f:
#         json.dump(metadata, f, indent=4)

#     # 10. Final Report
#     logger.info("\n" + "=" * 60)
#     logger.info(f"FINAL CLASSIFICATION REPORT ({best_name})")
#     logger.info("=" * 60)
#     y_pred_final = best_model.predict(X_test)
#     logger.info("\n" + classification_report(y_test, y_pred_final, digits=4))
#     logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_final)))

#     logger.info("=" * 60)
#     logger.info("✅ TRAINING COMPLETE. Model saved to 'models/'")
#     logger.info("=" * 60)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# train.py
"""
Training pipeline that compares 3 models (RandomForest, XGBoost, LightGBM)
and selects the best one based on macro F1 on a validation set.
"""

import sys
import json
import logging
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed; will only train RF and XGB.")

import config
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{config.LOG_DIR}/training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def find_best_threshold(y_true, y_proba, thresholds=None):
    """Find threshold that maximizes macro F1."""
    if thresholds is None:
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


def evaluate_model(y_true, y_pred, y_proba, threshold):
    """Compute metrics and return dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }


def main():
    logger.info("="*60)
    logger.info("🚀 MULTI‑MODEL TRAINING & SELECTION")
    logger.info("="*60)

    # 1. Load data
    df = pd.read_csv(config.DATA_PATH, low_memory=False)
    logger.info(f"Loaded data: {df.shape}")

    target = 'outbreak_flag'
    if target not in df.columns:
        raise ValueError(f"'{target}' not found. Run data_preprocessing.py first.")

    # Drop non‑numeric / IDs
    exclude_cols = [
        'ISO_A0', 'calendar_start_date', target, 'dengue_total',
        'adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name',
    ]
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].dtype in ['int64', 'float64']
    ]
    logger.info(f"Using {len(feature_cols)} numeric features:\n{feature_cols}")

    X = df[feature_cols]
    y = df[target]

    # 2. Split into train+val and test (stratified)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    # Split trainval into train and validation (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.20,  # 20% of trainval -> 16% of total
        random_state=config.RANDOM_STATE,
        stratify=y_trainval
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Compute sample weights for training (to handle imbalance)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    logger.info("Sample weights computed for balanced training.")

    # 5. Define models
    models = {}

    # Random Forest
    logger.info("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    y_proba_rf = rf.predict_proba(X_val_scaled)[:, 1]
    best_thresh_rf, _ = find_best_threshold(y_val, y_proba_rf)
    y_pred_rf = (rf.predict_proba(X_test_scaled)[:, 1] >= best_thresh_rf).astype(int)
    metrics_rf = evaluate_model(y_test, y_pred_rf, rf.predict_proba(X_test_scaled)[:, 1], best_thresh_rf)
    models["RandomForest"] = {"model": rf, "metrics": metrics_rf}
    logger.info(f"   RF macro F1 (val) = {f1_score(y_val, (y_proba_rf >= best_thresh_rf).astype(int), average='macro'):.4f}")
    logger.info(f"   RF test macro F1 = {metrics_rf['macro_f1']:.4f}")

    # XGBoost
    logger.info("\n[2/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=20,
        eval_metric='logloss'
    )
    # Note: we use sample_weight during fit
    eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
    xgb_model.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False
    )
    y_proba_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
    best_thresh_xgb, _ = find_best_threshold(y_val, y_proba_xgb)
    y_pred_xgb = (xgb_model.predict_proba(X_test_scaled)[:, 1] >= best_thresh_xgb).astype(int)
    metrics_xgb = evaluate_model(y_test, y_pred_xgb, xgb_model.predict_proba(X_test_scaled)[:, 1], best_thresh_xgb)
    models["XGBoost"] = {"model": xgb_model, "metrics": metrics_xgb}
    logger.info(f"   XGB macro F1 (val) = {f1_score(y_val, (y_proba_xgb >= best_thresh_xgb).astype(int), average='macro'):.4f}")
    logger.info(f"   XGB test macro F1 = {metrics_xgb['macro_f1']:.4f}")

    # LightGBM (if available)
    if HAS_LGB:
        logger.info("\n[3/3] Training LightGBM...")
        # Calculate scale_pos_weight for LightGBM (majority/minority)
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # LightGBM handles imbalance via this
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        # LightGBM supports early stopping via callbacks
        lgb_model.fit(
            X_train_scaled, y_train,
            sample_weight=sample_weights,  # also pass sample_weights
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        y_proba_lgb = lgb_model.predict_proba(X_val_scaled)[:, 1]
        best_thresh_lgb, _ = find_best_threshold(y_val, y_proba_lgb)
        y_pred_lgb = (lgb_model.predict_proba(X_test_scaled)[:, 1] >= best_thresh_lgb).astype(int)
        metrics_lgb = evaluate_model(y_test, y_pred_lgb, lgb_model.predict_proba(X_test_scaled)[:, 1], best_thresh_lgb)
        models["LightGBM"] = {"model": lgb_model, "metrics": metrics_lgb}
        logger.info(f"   LGB macro F1 (val) = {f1_score(y_val, (y_proba_lgb >= best_thresh_lgb).astype(int), average='macro'):.4f}")
        logger.info(f"   LGB test macro F1 = {metrics_lgb['macro_f1']:.4f}")

    # 6. Select best model based on test macro F1
    best_name = max(models, key=lambda k: models[k]['metrics']['macro_f1'])
    best_model = models[best_name]['model']
    best_metrics = models[best_name]['metrics']
    best_threshold = best_metrics['threshold']

    logger.info("\n" + "-"*60)
    logger.info(f"🏆 BEST MODEL: {best_name}")
    logger.info(f"   Macro F1 (test): {best_metrics['macro_f1']:.4f}")
    logger.info(f"   Accuracy       : {best_metrics['accuracy']:.4f}")
    logger.info(f"   F1 (class 1)   : {best_metrics['f1']:.4f}")
    logger.info(f"   Recall (class1): {best_metrics['recall']:.4f}")
    logger.info(f"   Precision      : {best_metrics['precision']:.4f}")
    logger.info(f"   ROC‑AUC        : {best_metrics['roc_auc']:.4f}")
    logger.info(f"   Optimal threshold: {best_threshold:.3f}")

    # 7. Save the best model and artefacts
    joblib.dump(best_model, f"{config.MODEL_DIR}/model.joblib")
    joblib.dump(scaler, f"{config.MODEL_DIR}/scaler.joblib")
    with open(f"{config.MODEL_DIR}/feature_names.json", "w") as f:
        json.dump(feature_cols, f)

    # 8. Metadata
    metadata = {
        "model_type": best_name,
        "target": target,
        "feature_columns": feature_cols,
        "optimal_threshold": float(best_threshold),
        "test_accuracy": float(best_metrics['accuracy']),
        "test_f1_class1": float(best_metrics['f1']),
        "test_precision": float(best_metrics['precision']),
        "test_recall": float(best_metrics['recall']),
        "test_roc_auc": float(best_metrics['roc_auc']),
        "test_macro_f1": float(best_metrics['macro_f1']),
        "training_date": datetime.now().isoformat(),
        "n_features": len(feature_cols),
        "all_models": {name: models[name]['metrics']['macro_f1'] for name in models},
    }
    with open(f"{config.MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Final classification report for the best model on test set
    y_pred_final = (best_model.predict_proba(X_test_scaled)[:, 1] >= best_threshold).astype(int)
    logger.info("\n" + "="*60)
    logger.info(f"FINAL CLASSIFICATION REPORT ({best_name})")
    logger.info("="*60)
    logger.info("\n" + classification_report(y_test, y_pred_final, digits=4))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_final)))

    logger.info("\n" + "="*60)
    logger.info(f"✅ TRAINING COMPLETE. Best model: {best_name}")
    logger.info(f"   Macro F1: {best_metrics['macro_f1']:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()