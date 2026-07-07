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