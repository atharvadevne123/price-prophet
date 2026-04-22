"""Airflow DAG for automated daily model retraining and drift-gated deployment."""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False

import logging

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "reflective-lantern",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

API_BASE = "http://price-prophet-api:8000"


def check_drift(**context):
    import requests
    resp = requests.get(f"{API_BASE}/drift", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    drift_rate = data.get("drift_rate", 0)
    n_recent = data.get("n_current_samples", 0)

    logger.info("Drift check: rate=%.3f n_recent=%d", drift_rate, n_recent)

    if n_recent < 20:
        context["task_instance"].xcom_push(key="needs_retrain", value=False)
        logger.info("Skipping retrain — insufficient recent predictions (%d)", n_recent)
        return

    needs_retrain = drift_rate > 0.3
    context["task_instance"].xcom_push(key="needs_retrain", value=needs_retrain)
    context["task_instance"].xcom_push(key="drifted_features", value=data.get("drifted_features", []))
    logger.info("Retrain needed: %s (drift_rate=%.3f)", needs_retrain, drift_rate)


def retrain_model(**context):
    needs_retrain = context["task_instance"].xcom_pull(key="needs_retrain")
    if not needs_retrain:
        logger.info("No retrain needed — skipping")
        return {"skipped": True}

    import requests
    resp = requests.post(f"{API_BASE}/train", json={"n_samples": 5000}, timeout=120)
    resp.raise_for_status()
    metrics = resp.json()
    logger.info("Retrain complete: %s", metrics)
    return metrics


def log_run_metrics(**context):
    import json
    from datetime import datetime
    run_date = datetime.utcnow().isoformat()
    metrics = context["task_instance"].xcom_pull(task_ids="retrain_model") or {}
    log_entry = {"date": run_date, "metrics": metrics, "drift_checked": True}
    logger.info("DAG run complete: %s", json.dumps(log_entry))


if HAS_AIRFLOW:
    with DAG(
        dag_id="price_prophet_retrain",
        default_args=DEFAULT_ARGS,
        description="Daily drift check and conditional model retraining for price-prophet",
        schedule="0 2 * * *",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["ml", "price-prophet", "retraining"],
    ) as dag:
        t_drift = PythonOperator(
            task_id="check_drift",
            python_callable=check_drift,
        )
        t_retrain = PythonOperator(
            task_id="retrain_model",
            python_callable=retrain_model,
        )
        t_log = PythonOperator(
            task_id="log_run_metrics",
            python_callable=log_run_metrics,
        )
        t_health = BashOperator(
            task_id="health_check",
            bash_command=f"curl -sf {API_BASE}/health || exit 1",
        )
        t_health >> t_drift >> t_retrain >> t_log
