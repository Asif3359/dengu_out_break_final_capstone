# Dengue Outbreak Predictor Backend

FastAPI service for predicting dengue outbreaks using XGBoost.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000