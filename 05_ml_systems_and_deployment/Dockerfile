# Minimal image for the fraud-scoring service.
# Build:  docker build -t fraud-scorer -f 05_ml_systems_and_deployment/Dockerfile .
# Run  :  docker run -p 8000:8000 fraud-scorer
FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/xgboost wheels (most prebuilt; keep small)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps -- subset of the repo-wide requirements to keep the image small.
# In production you'd pin exact versions and use a lock file.
RUN pip install --no-cache-dir \
        numpy pandas pyarrow scikit-learn xgboost joblib \
        fastapi uvicorn pydantic

# Copy code + the trained artifact.
COPY 05_ml_systems_and_deployment/01_inference_service.py /app/inference_service.py
COPY artifacts/ /app/artifacts/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
