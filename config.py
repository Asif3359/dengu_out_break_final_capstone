# config.py
import os

DATA_PATH = "data/merged_data.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model hyperparameters (Gradient Boosting)
GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "subsample": 0.8,
    "random_state": 42,
}

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Feature columns (will be overridden by the training script if using metadata)
# but kept for reference (these are the final columns after preprocessing)
# We'll keep it empty and let train.py read from saved metadata.
FEATURE_COLS = []  # not used if we load from metadata