import json
import logging
import os
import uuid

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from app.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")


def build_ensemble() -> VotingRegressor:
    estimators = [("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))]
    if HAS_XGB:
        estimators.append(("xgb", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="rmse")))
    if HAS_LGB:
        estimators.append(("lgb", LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)))
    if not HAS_XGB and not HAS_LGB:
        estimators.append(("gbm", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)))
    return VotingRegressor(estimators=estimators)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", build_ensemble()),
    ])


def train_model(X: np.ndarray, y: np.ndarray) -> dict:
    pipeline = build_pipeline()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    neg_mse_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse_scores)

    pipeline.fit(X, y)

    run_id = str(uuid.uuid4())[:8]
    metrics = {
        "run_id": run_id,
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": FEATURE_NAMES,
        "estimators": [name for name, _ in build_ensemble().estimators],
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model trained: run_id=%s rmse=%.4f±%.4f", run_id, rmse_scores.mean(), rmse_scores.std())
    return metrics


def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        logger.warning("No model found at %s — training on synthetic data", MODEL_PATH)
        from app.features import generate_synthetic_training_data
        X, y = generate_synthetic_training_data()
        train_model(X, y)
    return joblib.load(MODEL_PATH)


def predict(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.predict(X)


def load_metrics() -> dict:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def optimize_price(
    pipeline: Pipeline,
    base_features: np.ndarray,
    cost: float,
    price_range: tuple = (0.7, 1.5),
    n_steps: int = 20,
) -> dict:
    best_price_mult = 1.0
    best_revenue = -1.0

    for mult in np.linspace(price_range[0], price_range[1], n_steps):
        features = base_features.copy()
        base_price = features[0]
        new_price = base_price * mult
        features[0] = new_price
        features[2] = new_price / max(features[1], 0.01)

        demand = float(predict(pipeline, features.reshape(1, -1))[0])
        profit = (new_price - cost) * demand

        if profit > best_revenue:
            best_revenue = profit
            best_price_mult = mult
            best_demand = demand
            best_new_price = new_price

    return {
        "optimized_price": round(best_new_price, 2),
        "expected_demand": round(best_demand, 2),
        "expected_profit": round(best_revenue, 2),
        "price_multiplier": round(best_price_mult, 3),
    }
