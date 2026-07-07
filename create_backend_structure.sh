#!/bin/bash
# create_backend_structure.sh
# Creates the enterprise‑grade backend folder structure with placeholder files.

set -e  # exit on error

echo "🚀 Creating backend folder structure..."

# Create directories
mkdir -p backend/app/core
mkdir -p backend/app/api/v1/endpoints
mkdir -p backend/app/models
mkdir -p backend/app/services
mkdir -p backend/app/utils
mkdir -p backend/tests

# Create __init__.py files
touch backend/app/__init__.py
touch backend/app/core/__init__.py
touch backend/app/api/__init__.py
touch backend/app/api/v1/__init__.py
touch backend/app/api/v1/endpoints/__init__.py
touch backend/app/models/__init__.py
touch backend/app/services/__init__.py
touch backend/app/utils/__init__.py
touch backend/tests/__init__.py

# Create main code files (empty)
touch backend/app/main.py
touch backend/app/core/config.py
touch backend/app/core/dependencies.py
touch backend/app/api/v1/endpoints/health.py
touch backend/app/api/v1/endpoints/predict.py
touch backend/app/api/v1/endpoints/features.py
touch backend/app/models/schemas.py
touch backend/app/services/predictor.py
touch backend/app/utils/logging.py
touch backend/tests/test_api.py

# Create root-level files (empty)
touch backend/.env.example
touch backend/requirements.txt
touch backend/Dockerfile
touch backend/docker-compose.yml
touch backend/README.md

echo "✅ Backend folder structure created successfully!"
echo "📁 Structure:"
tree backend/