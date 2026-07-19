# Development Dockerfile for backend FastAPI application
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# # deployment 

# FROM python:3.11-slim

# WORKDIR /app

# COPY backend/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY backend/ .

# COPY models/ /app/models/

# RUN ls -la /app/models || echo "Models directory not found"


# ENV PYTHONPATH=/app
# ENV MODEL_DIR=/app/models

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]