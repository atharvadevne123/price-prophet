import numpy as np
import pytest

from app.features import generate_synthetic_training_data, FEATURE_NAMES
from app.model import build_pipeline, predict, optimize_price, train_model


@pytest.fixture(scope="module")
def trained_pipeline():
    X, y = generate_synthetic_training_data(n_samples=300)
    train_model(X, y)
    from app.model import load_model
    return load_model()


def test_pipeline_builds():
    pipeline = build_pipeline()
    assert pipeline is not None
    steps = dict(pipeline.steps)
    assert "scaler" in steps
    assert "model" in steps


def test_train_returns_metrics():
    X, y = generate_synthetic_training_data(n_samples=200)
    metrics = train_model(X, y)
    assert "run_id" in metrics
    assert "rmse_mean" in metrics
    assert metrics["rmse_mean"] >= 0
    assert metrics["n_features"] == len(FEATURE_NAMES)


def test_predict_shape(trained_pipeline):
    X, _ = generate_synthetic_training_data(n_samples=10)
    preds = predict(trained_pipeline, X)
    assert preds.shape == (10,)


def test_predict_non_negative(trained_pipeline):
    X, _ = generate_synthetic_training_data(n_samples=50)
    preds = predict(trained_pipeline, X)
    assert (preds >= 0).mean() > 0.8


def test_optimize_price_returns_dict(trained_pipeline):
    X, _ = generate_synthetic_training_data(n_samples=1)
    features = X[0]
    cost = features[0] * 0.6
    result = optimize_price(trained_pipeline, features, cost)
    assert "optimized_price" in result
    assert "expected_demand" in result
    assert "expected_profit" in result
    assert result["optimized_price"] > 0


def test_cross_val_5_fold():
    from sklearn.model_selection import KFold, cross_val_score
    X, y = generate_synthetic_training_data(n_samples=300)
    pipeline = build_pipeline()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
    assert len(scores) == 5
    rmse_scores = np.sqrt(-scores)
    assert all(r >= 0 for r in rmse_scores)
