import numpy as np

from app.features import generate_synthetic_training_data
from app.monitoring import (
    compute_drift,
    compute_feature_drift,
    load_reference_stats,
    prediction_health_check,
    save_reference_stats,
)


def test_compute_drift_no_drift():
    ref = list(np.random.normal(0, 1, 100))
    cur = list(np.random.normal(0, 1, 100))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert isinstance(result["drift_detected"], bool)


def test_compute_drift_with_drift():
    ref = list(np.random.normal(0, 1, 200))
    cur = list(np.random.normal(5, 1, 200))  # clearly shifted
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    result = compute_drift([1, 2], [3])
    assert "error" in result
    assert result["drift_detected"] is False


def test_feature_drift_no_drift():
    X_ref, _ = generate_synthetic_training_data(n_samples=200)
    X_cur, _ = generate_synthetic_training_data(n_samples=100)
    result = compute_feature_drift(X_ref, X_cur)
    assert "feature_results" in result
    assert "drift_detected" in result
    assert "drift_rate" in result
    assert 0.0 <= result["drift_rate"] <= 1.0


def test_prediction_health_check_normal():
    preds = [10.5, 20.3, 15.0, 30.2, 5.1]
    health = prediction_health_check(preds)
    assert health["count"] == 5
    assert health["negative_pct"] == 0.0
    assert "mean" in health
    assert "std" in health


def test_prediction_health_check_empty():
    result = prediction_health_check([])
    assert result["status"] == "no_data"


def test_save_load_reference_stats(tmp_path):
    import os
    os.environ["REFERENCE_STATS_PATH"] = str(tmp_path / "ref_stats.json")
    X, _ = generate_synthetic_training_data(n_samples=100)
    save_reference_stats(X)
    stats = load_reference_stats()
    assert "mean" in stats
    assert "std" in stats
    assert stats["n_samples"] == 100
