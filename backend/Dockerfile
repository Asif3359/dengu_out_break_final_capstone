FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies (relative to backend folder)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend code
COPY backend/ .

# Copy the models from the parent directory (since build context will be the repo root)
COPY models/ /app/models

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]