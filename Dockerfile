FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY dags/ ./dags/

ENV DATABASE_URL=sqlite:///./price_prophet.db
ENV MODEL_PATH=model.joblib
ENV METRICS_PATH=metrics.json
ENV REFERENCE_STATS_PATH=reference_stats.json
ENV FAISS_INDEX_PATH=faiss_index.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
