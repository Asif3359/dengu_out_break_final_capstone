# Dengue Outbreak Prediction System

## Full‑Stack Machine Learning Solution for Early Warning of Dengue Epidemics

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Running with Docker (Recommended)](#running-with-docker-recommended)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides an **end‑to‑end machine learning pipeline** for predicting dengue outbreaks in the WHO South‑East Asia Region (SEARO). It combines:

- **OpenDengue** surveillance data (10 countries, 1981–2025)
- **NASA POWER** meteorological data (temperature & precipitation)
- **Feature engineering** with lagged variables, rolling statistics, and cyclical encoding
- **XGBoost** classification model (macro F1 = 0.695, outbreak recall = 95.8%)
- **FastAPI** backend with enterprise‑grade structure
- **Next.js** dashboard with ShadCN UI for interactive predictions

The system is designed for public health officials and researchers to assess outbreak risk in real time, supporting early intervention and resource allocation.

---

## Features

- **Data Pipeline** – automated preprocessing, weather API integration, and feature engineering
- **Multi‑Model Comparison** – Random Forest, XGBoost, LightGBM with hyperparameter tuning
- **Class‑Imbalance Handling** – sample weighting and threshold optimization
- **Deep Learning Benchmark** – LSTM & Transformer models with SMOTE (experimental)
- **Interactive Dashboard** – user‑friendly Next.js UI with preset scenarios
- **RESTful API** – FastAPI with /predict and /predict_batch endpoints
- **Enterprise‑Ready** – modular architecture, Docker, comprehensive logging, and tests
- **Interpretability** – SHAP analysis (optional) and feature importance

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Next.js Frontend                       │
│                   (Dashboard with ShadCN UI)                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  /predict    │  │  /health     │  │  /feature_names     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ Predictor    │  │ Scaler       │                           │
│  │ (XGBoost)    │  │ (Standard)   │                           │
│  └──────────────┘  └──────────────┘                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Artifacts (models/)                   │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ model.joblib │ scaler.joblib │ metadata.json │ feature_names ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Next.js (App Router), TypeScript, Tailwind CSS, ShadCN UI, Recharts |
| **Backend** | FastAPI, Python 3.11+, Pydantic, Uvicorn |
| **ML** | XGBoost, Scikit‑learn, Pandas, NumPy, Joblib |
| **Data** | OpenDengue (CSV), NASA POWER API (JSON), Parquet cache |
| **DevOps** | Docker, Docker Compose, Git |
| **Testing** | Pytest, HTTPX |

---

## Prerequisites

- **Node.js** ≥ 18.x
- **Python** ≥ 3.11 (recommended) – if using local development
- **Docker** and **Docker Compose** (optional, recommended)
- **Git**
- (Optional) Conda for Python environment management

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/Asif3359/dengu_out_break_final_capstone.git
cd dengu_out_break_final_capstone
```

---

### Backend Setup

#### Option A: Local Development (with Conda)

1. **Create and activate a conda environment** (Python 3.11):
   ```bash
   conda create -n dengue_dev python=3.11
   conda activate dengue_dev
   ```

2. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Copy environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` if needed (defaults usually work).

4. **Run the API server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Verify** – visit `http://localhost:8000/docs` for interactive Swagger UI.

#### Option B: Using Docker (Recommended)

```bash
cd backend
docker compose up --build -d
```

The API will be available at `http://localhost:8000`.

---

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd dashboard
   npm install
   ```

2. **Set environment variables**:
   Create `.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

4. **Access the dashboard** – open `http://localhost:3000`.

---

### Running with Docker (Complete Stack)

If you have both the backend and frontend Dockerised, you can run the entire stack with a single command (you may need to adapt the `docker-compose.yml` in the root or run separately). For now, we run them independently:

```bash
# Terminal 1 – Backend
cd backend
docker compose up

# Terminal 2 – Frontend
cd dashboard
npm run dev
```

---

## Environment Variables

### Backend (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Path to the models folder | `../models` |
| `MODEL_FILENAME` | Name of the serialised model | `model.joblib` |
| `SCALER_FILENAME` | Name of the scaler file | `scaler.joblib` |
| `FEATURE_NAMES_FILE` | File listing feature order | `feature_names.json` |
| `METADATA_FILE` | File with model metadata | `metadata.json` |
| `DEFAULT_THRESHOLD` | Fallback classification threshold | `0.5` |
| `CORS_ORIGINS` | Comma‑separated allowed origins | `["http://localhost:3000"]` |

### Frontend (`.env.local`)

| Variable | Description | Required |
|----------|-------------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL | Yes |

---

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests (if added)

```bash
cd dashboard
npm test   # (if configured)
```

---

## Project Structure

```
outbreak_final/
├── backend/                        # FastAPI backend
│   ├── app/                        # Application package
│   │   ├── api/v1/endpoints/       # Routers (health, predict, features)
│   │   ├── core/                   # Config & dependencies
│   │   ├── models/                 # Pydantic schemas
│   │   ├── services/               # Predictor service (loads model)
│   │   ├── utils/                  # Logging, etc.
│   │   └── main.py                 # App factory
│   ├── tests/                      # Unit/integration tests
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── README.md
├── dashboard/                      # Next.js frontend
│   ├── src/
│   │   ├── app/                    # Layout & pages
│   │   ├── components/             # UI components (ShadCN)
│   │   ├── hooks/                  # Custom hooks (usePrediction)
│   │   ├── services/               # API client (axios)
│   │   ├── types/                  # TypeScript interfaces
│   │   └── utils/                  # Constants, helpers
│   ├── public/
│   ├── package.json
│   └── next.config.ts
├── models/                         # Trained model artifacts
│   ├── model.joblib
│   ├── scaler.joblib
│   ├── feature_names.json
│   └── metadata.json
├── data/                           # Raw & processed data
│   ├── filtered_data_SEARO_*.csv   # Raw dengue data
│   ├── merged_data.csv             # Feature‑engineered dataset
│   └── weather_cache/              # Cached NASA POWER responses
├── figures/                        # EDA visualisations
├── logs/                           # Training logs
├── config.py                       # Training config
├── data_preprocessing.py           # Preprocessing pipeline
├── eda_analysis.py                 # EDA script
├── train.py                        # Multi‑model training & selection
├── train_lstm.py                   # LSTM/Transformer training
├── predict.py                      # CLI prediction script
├── environment.yml                 # Conda env export
├── .gitignore
├── .env.example
└── README.md                       # This file
```

---

## API Documentation

Once the backend is running, Swagger UI is available at `/docs`. The main endpoints are:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/api/v1/feature_names` | Returns list of required features |
| `POST` | `/api/v1/predict` | Single prediction |
| `POST` | `/api/v1/predict_batch` | Batch predictions (array of feature objects) |

### Example Request

```json
POST /api/v1/predict
Content-Type: application/json

{
  "features": {
    "month": 6,
    "year": 2024,
    "temperature_c": 28.5,
    "precipitation_mm": 5.2,
    ...
  }
}
```

### Example Response

```json
{
  "outbreak_probability": 0.892,
  "prediction": 1,
  "threshold_used": 0.33,
  "model_type": "XGBoost"
}
```

---

## Deployment

### Backend

1. Build the Docker image:
   ```bash
   docker build -t dengue-api .
   ```
2. Run with your preferred orchestrator (e.g., `docker run -p 8000:8000 dengue-api`).
3. For production, set `CORS_ORIGINS` and use a reverse proxy (Nginx) with HTTPS.

### Frontend

1. Build the Next.js app:
   ```bash
   npm run build
   ```
2. Export static files (optional) or deploy to Vercel, Netlify, or a Node.js host:
   ```bash
   npm start
   ```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, discuss first.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- OpenDengue Project for providing the surveillance data.
- NASA POWER for weather data API.
- The SEARO countries for their public health data.
- Contributors and open‑source libraries used.

---

## Contact

For questions or support, please open an issue on GitHub or contact the maintainer.

---

**Happy Predicting!**