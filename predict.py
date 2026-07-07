# predict.py
import sys
import json
import logging
import warnings

import joblib
import pandas as pd
import numpy as np

import config

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OutbreakPredictor:
    def __init__(self, model_dir="models"):
        """Load all artifacts from the model directory."""
        self.model_dir = model_dir

        # Load artifacts
        self.model = joblib.load(f"{model_dir}/model.joblib")
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        self.encoder = joblib.load(f"{model_dir}/encoder.joblib")

        # Load metadata
        with open(f"{model_dir}/metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata["feature_columns"]
        self.numeric_columns = self.metadata["numeric_columns"]
        self.optimal_threshold = self.metadata["optimal_threshold"]

        logger.info(f"Predictor initialized. Model: {self.metadata['model_name']}")
        logger.info(f"Threshold: {self.optimal_threshold:.4f}")

    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Check if the input dataframe has all required columns."""
        required_cols = ["ISO_A0", "year", "month", "dayofyear", "weekofyear",
                         "temperature_c", "precipitation_mm"]

        # We also need the lag features, but they are generated automatically in the pipeline.
        # For simplicity, we expect the user to provide the raw features.
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def predict(self, df: pd.DataFrame, apply_threshold: bool = True):
        """
        Make predictions on new data.

        Args:
            df: DataFrame with raw features (ISO_A0, year, month, dayofyear, weekofyear,
                temperature_c, precipitation_mm). Lags and rolling stats will be auto-generated.
            apply_threshold: If True, returns binary (0/1) using optimal threshold.
                              If False, returns probabilities.
        Returns:
            predictions: list of ints (0 or 1) or floats (probabilities).
        """
        # 1. Validate
        self._validate_input(df)

        # 2. Feature engineering (same as training)
        # Create a copy to avoid modifying original
        X = df.copy()

        # Add required features (if missing, fill with 0. In production, you might handle differently)
        # Actually, the safest is to re-run the exact engineering.
        # But to keep predict.py self-contained, we replicate the logic.
        # For efficiency, we can just check if the engineered features exist; if not, compute them.

        # Ensure lag columns exist (we need dengue_total for lags, but during prediction we don't have it!)
        # Wait! In a real outbreak prediction, you DO have past dengue cases to compute lags.
        # The user will need to provide 'dengue_total' with historical values to compute lags.
        # I will assume the incoming dataframe has 'dengue_total' as well for lag calculation.
        if "dengue_total" not in X.columns:
            raise ValueError("Input must contain 'dengue_total' for lag feature engineering.")

        # Compute lags
        for i in config.LAG_DAYS:
            X[f"lag_{i}"] = X.groupby("ISO_A0")["dengue_total"].shift(i)

        # Compute rolling stats (note: rolling needs order, ensure sorted)
        X = X.sort_values(["ISO_A0", "year", "month", "dayofyear"])  # rough sort
        for window in config.ROLLING_WINDOWS:
            X[f"temp_rolling_mean_{window}"] = X.groupby("ISO_A0")["temperature_c"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            X[f"precip_rolling_std_{window}"] = X.groupby("ISO_A0")["precipitation_mm"].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

        # Cyclical
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        X["dayofyear_sin"] = np.sin(2 * np.pi * X["dayofyear"] / 365)
        X["dayofyear_cos"] = np.cos(2 * np.pi * X["dayofyear"] / 365)
        X["temp_precip_interaction"] = X["temperature_c"] * X["precipitation_mm"]

        # 3. Encode ISO_A0
        X["ISO_A0_encoded"] = self.encoder.transform(X["ISO_A0"])

        # 4. Scale numeric columns
        numeric_cols = [col for col in self.numeric_columns if col != "ISO_A0"]
        # Ensure all numeric cols exist, fill NaN with 0 or mean? In production, we should handle.
        for col in numeric_cols:
            if col not in X.columns:
                X[col] = 0  # fallback, but ideally the engineering covers it.
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        # 5. Select final features in correct order
        final_X = X[self.feature_columns]
        final_X = final_X.dropna()  # drop rows with NaN (e.g., first few lags)

        if final_X.empty:
            raise ValueError("All rows resulted in NaN after feature engineering. Check input data.")

        # 6. Predict
        if apply_threshold:
            proba = self.model.predict_proba(final_X)[:, 1]
            return (proba >= self.optimal_threshold).astype(int).tolist()
        else:
            return self.model.predict_proba(final_X)[:, 1].tolist()


def main():
    """CLI entry point: python predict.py --input data.csv --output predictions.csv"""
    import argparse

    parser = argparse.ArgumentParser(description="Predict outbreaks using trained model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV file.")
    parser.add_argument("--probabilities", action="store_true", help="Output probabilities instead of binary.")
    args = parser.parse_args()

    # Load input
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {df.shape} rows for prediction.")

    # Predict
    predictor = OutbreakPredictor()
    predictions = predictor.predict(df, apply_threshold=not args.probabilities)

    # Save
    df_out = pd.DataFrame({
        "prediction": predictions,
        "threshold_used": predictor.optimal_threshold
    })
    df_out.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()