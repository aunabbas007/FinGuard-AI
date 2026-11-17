FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY src ./src
COPY web ./web
# Recreate the deterministic synthetic training set and model inside the image.
# This avoids committing binary model artifacts or requiring a manual model upload.
RUN python src/data/generate_transactions_v2.py \
 && python src/features/build_features.py \
 && python src/graph/build_graph_features.py \
 && python src/features/merge_graph_features.py \
 && python src/models/evaluate_models.py
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["sh","-c","uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
