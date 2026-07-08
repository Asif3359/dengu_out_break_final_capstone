#!/usr/bin/env python
# train_lstm.py
"""
Deep learning time‑series models for dengue outbreak prediction.
- LSTM (Bidirectional) with class weights and SMOTE
- Transformer (self‑attention) with SMOTE
- Threshold optimization and SHAP interpretability
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Add, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping
import shap

warnings.filterwarnings("ignore")

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/lstm_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ===========================================================================
# 1. LOAD AND PREPARE DATA
# ===========================================================================
def load_and_prepare_data(data_path="data/merged_data.csv", lookback=12):
    """
    Load data, create sequences per country, and apply SMOTE only on training set.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols
    """
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded data: {df.shape}")

    # Ensure datetime and sort per country
    df['calendar_start_date'] = pd.to_datetime(df['calendar_start_date'])
    df = df.sort_values(['ISO_A0', 'calendar_start_date']).reset_index(drop=True)

    # Target
    target = 'outbreak_flag'
    if target not in df.columns:
        raise ValueError(f"'{target}' not found. Run data_preprocessing.py first.")

    # Features: exclude non‑numeric and target
    exclude_cols = [
        'ISO_A0', 'calendar_start_date', 'dengue_total', target,
        'adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name',
    ]
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].dtype in ['int64', 'float64']
    ]
    logger.info(f"Using {len(feature_cols)} numeric features")

    # Create sequences per country
    countries = df['ISO_A0'].unique()
    X_seq, y_seq = [], []
    scaler = StandardScaler()

    for country in countries:
        country_df = df[df['ISO_A0'] == country].sort_values('calendar_start_date')
        features_scaled = scaler.fit_transform(country_df[feature_cols].values)
        targets = country_df[target].values

        for i in range(lookback, len(features_scaled)):
            X_seq.append(features_scaled[i-lookback:i])
            y_seq.append(targets[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int32)

    logger.info(f"Total sequences: {X_seq.shape[0]}, lookback={lookback}, features={X_seq.shape[2]}")

    # Split chronologically (80% train, 10% val, 10% test)
    n = len(X_seq)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    X_train_raw, y_train_raw = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]

    logger.info(f"Before SMOTE - Train: {X_train_raw.shape}, class counts: {np.bincount(y_train_raw)}")
    logger.info(f"Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- Apply SMOTE on training sequences ----
    n_samples, lookback, n_features = X_train_raw.shape
    X_train_flat = X_train_raw.reshape(n_samples, lookback * n_features)
    smote = SMOTE(random_state=42)
    X_train_flat_res, y_train_res = smote.fit_resample(X_train_flat, y_train_raw)
    X_train = X_train_flat_res.reshape(-1, lookback, n_features)
    y_train = y_train_res
    logger.info(f"After SMOTE - Train: {X_train.shape}, class counts: {np.bincount(y_train)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols


# ===========================================================================
# 2. BUILD MODELS
# ===========================================================================
def build_lstm(input_shape):
    """Bidirectional LSTM with dropout."""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_transformer(input_shape, d_model=64, n_heads=4, ff_dim=128, dropout=0.1):
    """Improved Transformer with feed‑forward and residual connections."""
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    # Learnable positional embedding
    pos_emb = tf.keras.layers.Embedding(input_shape[0], d_model)(tf.range(input_shape[0]))
    x = x + tf.expand_dims(pos_emb, axis=0)

    # Multi‑head attention
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # Feed‑forward
    ffn = Sequential([
        Dense(ff_dim, activation='relu'),
        Dense(d_model)
    ])
    ffn_out = ffn(x)
    x = Add()([x, ffn_out])
    x = LayerNormalization()(x)

    # Global pooling and classification
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ===========================================================================
# 3. TRAINING, EVALUATION & THRESHOLD TUNING
# ===========================================================================
def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """
    Train with class weights, early stopping, then find optimal threshold on validation
    and evaluate on test with that threshold.
    """
    logger.info(f"\nTraining {model_name}...")

    # Compute class weights for training
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weight_dict}")

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0
    )

    # ---- Threshold optimisation on validation set ----
    val_proba = model.predict(X_val, verbose=0).flatten()
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_macro_f1 = 0.0
    for thresh in thresholds:
        y_pred_tmp = (val_proba >= thresh).astype(int)
        f1_macro = f1_score(y_val, y_pred_tmp, average='macro')
        if f1_macro > best_macro_f1:
            best_macro_f1 = f1_macro
            best_thresh = thresh
    logger.info(f"Optimal threshold (macro F1) = {best_thresh:.3f}")

    # ---- Apply threshold to test set ----
    test_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (test_proba >= best_thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, test_proba)

    logger.info(f"{model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info(f"\nClassification Report ({model_name}):\n{classification_report(y_test, y_pred, digits=4)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return model, history, {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "threshold": best_thresh
    }


# ===========================================================================
# 4. SHAP INTERPRETABILITY
# ===========================================================================
def shap_explain(model, X_sample, feature_names, model_type='lstm'):
    """
    Use KernelExplainer to compute SHAP values and print top features.
    """
    logger.info("\nApplying SHAP for interpretability...")
    # Reduce background size for speed
    if len(X_sample) > 100:
        background = X_sample[np.random.choice(X_sample.shape[0], 50, replace=False)]
    else:
        background = X_sample

    # KernelExplainer (model‑agnostic)
    explainer = shap.KernelExplainer(model.predict, background)
    # Explain first 5 test samples
    test_samples = X_sample[:5]
    shap_values = explainer.shap_values(test_samples, nsamples=100)

    # For binary classification, shap_values is a list [shap_class0, shap_class1]
    # We use class 1 (outbreak)
    shap_vals = shap_values[1]  # shape: (n_samples, lookback, n_features)
    # Average absolute SHAP over samples and time steps
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 1))
    feature_importance = dict(zip(feature_names, mean_abs_shap))
    sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 10 features by mean |SHAP| (averaged over time):")
    for feat, imp in sorted_imp[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    np.save("shap_values.npy", shap_values)
    logger.info("SHAP values saved to shap_values.npy")


# ===========================================================================
# 5. MAIN
# ===========================================================================
def main():
    logger.info("="*60)
    logger.info("   LSTM & TRANSFORMER TRAINING WITH SMOTE, WEIGHTS & THRESHOLD")
    logger.info("="*60)

    LOOKBACK = 12
    DATA_PATH = "data/merged_data.csv"

    # Load and prepare data (with SMOTE)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = load_and_prepare_data(DATA_PATH, LOOKBACK)

    input_shape = (LOOKBACK, len(feature_cols))

    # --- LSTM ---
    lstm_model = build_lstm(input_shape)
    lstm_model, _, lstm_metrics = train_and_evaluate(
        lstm_model, X_train, y_train, X_val, y_val, X_test, y_test, "LSTM"
    )

    # --- Transformer ---
    transformer_model = build_transformer(input_shape)
    transformer_model, _, transformer_metrics = train_and_evaluate(
        transformer_model, X_train, y_train, X_val, y_val, X_test, y_test, "Transformer"
    )

    # --- Model Comparison ---
    logger.info("\n" + "-"*60)
    logger.info("MODEL COMPARISON")
    logger.info("-"*60)
    logger.info(f"LSTM        - F1: {lstm_metrics['f1']:.4f}, ROC-AUC: {lstm_metrics['roc_auc']:.4f}, Threshold: {lstm_metrics['threshold']:.3f}")
    logger.info(f"Transformer - F1: {transformer_metrics['f1']:.4f}, ROC-AUC: {transformer_metrics['roc_auc']:.4f}, Threshold: {transformer_metrics['threshold']:.3f}")

    # --- Pick Best ---
    if lstm_metrics['f1'] >= transformer_metrics['f1']:
        best_model = lstm_model
        best_name = "LSTM"
        best_metrics = lstm_metrics
    else:
        best_model = transformer_model
        best_name = "Transformer"
        best_metrics = transformer_metrics

    logger.info(f"   BEST MODEL: {best_name} (F1: {best_metrics['f1']:.4f})")

    # Save best model
    os.makedirs("models", exist_ok=True)
    best_model.save(f"models/{best_name.lower()}_model.keras")
    logger.info(f"Saved {best_name} model to models/{best_name.lower()}_model.keras")

    # Save metadata
    metadata = {
        "model_type": best_name,
        "feature_columns": feature_cols,
        "lookback": LOOKBACK,
        "optimal_threshold": float(best_metrics['threshold']),
        "test_accuracy": float(best_metrics['accuracy']),
        "test_f1": float(best_metrics['f1']),
        "test_precision": float(best_metrics['precision']),
        "test_recall": float(best_metrics['recall']),
        "test_roc_auc": float(best_metrics['roc_auc']),
        "training_date": datetime.now().isoformat(),
        "n_features": len(feature_cols),
    }
    with open(f"models/{best_name.lower()}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to models/{best_name.lower()}_metadata.json")

    # SHAP explanation on best model (use a small subset of test data)
    shap_explain(best_model, X_test[:20], feature_cols, best_name.lower())

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE.")
    logger.info("="*60)


if __name__ == "__main__":
    main()