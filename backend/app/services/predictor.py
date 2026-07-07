import joblib
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple

class PredictorService:
    def __init__(self, model_dir: str, model_filename: str, scaler_filename: str,
                 feature_names_file: str, metadata_file: str, default_threshold: float):
        self.model_dir = model_dir
        self.model = joblib.load(os.path.join(model_dir, model_filename))
        self.scaler = joblib.load(os.path.join(model_dir, scaler_filename))
        with open(os.path.join(model_dir, feature_names_file), "r") as f:
            self.feature_names = json.load(f)
        with open(os.path.join(model_dir, metadata_file), "r") as f:
            self.metadata = json.load(f)
        self.optimal_threshold = self.metadata.get("optimal_threshold", default_threshold)
        self.model_type = self.metadata.get("model_type", "XGBoost")

    def predict_proba(self, features_dict: Dict[str, float]) -> Tuple[float, int]:
        # Extract in correct order
        features = [features_dict.get(name, 0.0) for name in self.feature_names]
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0, 1]
        pred = int(proba >= self.optimal_threshold)
        return proba, pred

    def predict_batch(self, items: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            try:
                proba, pred = self.predict_proba(item)
                results.append({
                    "input": item,
                    "probability": float(proba),
                    "prediction": pred
                })
            except Exception as e:
                results.append({"input": item, "error": str(e)})
        return results