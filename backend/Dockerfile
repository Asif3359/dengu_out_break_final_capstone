# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ---------- Install Python dependencies ----------
# Copy requirements from the backend folder
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy backend application code ----------
COPY backend/ .

# ---------- Copy the trained models from the repository root ----------
# Since the build context is the repo root, the `models/` folder is available.
COPY models/ /app/models/

# (Optional) Debug: list the copied models to verify they are present
RUN ls -la /app/models || echo "Models directory not found"

# ---------- Set environment variables ----------
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

# ---------- Run the FastAPI application ----------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]