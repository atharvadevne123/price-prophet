import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import DriftLog, ModelMetrics, Prediction, get_db, init_db
from app.features import engineer_features, generate_synthetic_training_data
from app.model import load_metrics, load_model, optimize_price, predict, train_model
from app.monitoring import (
    compute_feature_drift,
    prediction_health_check,
    save_reference_stats,
)
from app.retrieval import get_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_model = None
_reference_X: Optional[np.ndarray] = None
_recent_X: List[np.ndarray] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _reference_X
    init_db()
    _model = load_model()
    X_ref, _ = generate_synthetic_training_data(n_samples=500)
    _reference_X = X_ref
    save_reference_stats(X_ref)
    get_index()
    logger.info("price-prophet startup complete")
    yield
    logger.info("price-prophet shutdown")


app = FastAPI(
    title="price-prophet",
    description="E-commerce price optimization and demand forecasting engine",
    version="1.0.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    product_id: str = Field(..., example="PROD-001")
    base_price: float = Field(..., gt=0, example=99.99)
    competitor_price: Optional[float] = Field(None, example=105.0)
    category: str = Field("other", example="electronics")
    stock_level: Optional[float] = Field(100, example=150)
    cost: Optional[float] = Field(None, example=60.0)
    date: Optional[str] = Field(None, example="2024-12-15")
    historical_demand_7d: Optional[float] = Field(50, example=75)
    historical_demand_30d: Optional[float] = Field(200, example=280)
    days_since_last_promotion: Optional[float] = Field(30, example=14)


class TrainRequest(BaseModel):
    n_samples: int = Field(2000, ge=100, le=50000)


class SimilarRequest(BaseModel):
    base_price: float = Field(..., gt=0)
    category: str = "other"
    competitor_price: Optional[float] = None
    k: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/forecast")
def forecast(req: ForecastRequest, db: Session = Depends(get_db)):
    if _model is None:
        raise HTTPException(503, "Model not ready")

    data = req.model_dump()
    features = engineer_features(data)
    demand = float(predict(_model, features.reshape(1, -1))[0])
    demand = max(demand, 0.0)

    cost = float(req.cost or req.base_price * 0.6)
    opt = optimize_price(_model, features, cost)

    _recent_X.append(features)
    if len(_recent_X) > 1000:
        _recent_X.pop(0)

    pred_row = Prediction(
        product_id=req.product_id,
        category=req.category,
        predicted_demand=round(demand, 2),
        optimized_price=opt["optimized_price"],
        confidence=min(1.0, demand / 100),
        features_used=data,
    )
    db.add(pred_row)
    db.commit()

    return {
        "product_id": req.product_id,
        "predicted_demand": round(demand, 2),
        "optimized_price": opt["optimized_price"],
        "expected_profit": opt["expected_profit"],
        "price_multiplier": opt["price_multiplier"],
        "recommendation": "increase" if opt["price_multiplier"] > 1.05 else "decrease" if opt["price_multiplier"] < 0.95 else "hold",
    }


@app.post("/train")
def train(req: TrainRequest, db: Session = Depends(get_db)):
    global _model, _reference_X
    X, y = generate_synthetic_training_data(n_samples=req.n_samples)
    metrics = train_model(X, y)
    _model = load_model()
    _reference_X = X
    save_reference_stats(X)

    metrics_row = ModelMetrics(
        run_id=metrics["run_id"],
        auc_mean=0.0,
        auc_std=0.0,
        rmse=metrics["rmse_mean"],
        n_features=metrics["n_features"],
        n_samples=metrics["n_samples"],
    )
    db.add(metrics_row)
    db.commit()

    return {
        "status": "trained",
        "run_id": metrics["run_id"],
        "rmse_mean": round(metrics["rmse_mean"], 4),
        "rmse_std": round(metrics["rmse_std"], 4),
        "n_samples": metrics["n_samples"],
        "estimators": metrics["estimators"],
    }


@app.get("/drift")
def drift(db: Session = Depends(get_db)):
    if _reference_X is None:
        raise HTTPException(503, "Reference data not available")
    if len(_recent_X) < 10:
        return {"message": "Not enough recent predictions for drift analysis", "n_recent": len(_recent_X)}

    current_matrix = np.vstack(_recent_X)
    result = compute_feature_drift(_reference_X, current_matrix)

    for fname, fdata in result["feature_results"].items():
        if "error" not in fdata:
            db.add(DriftLog(
                feature_name=fname,
                ks_statistic=fdata["ks_statistic"],
                p_value=fdata["p_value"],
                drift_detected=int(fdata["drift_detected"]),
            ))
    db.commit()

    return {
        "drift_detected": result["drift_detected"],
        "drift_rate": result["drift_rate"],
        "drifted_features": result["drifted_features"],
        "n_reference_samples": _reference_X.shape[0],
        "n_current_samples": current_matrix.shape[0],
    }


@app.post("/similar")
def similar_products(req: SimilarRequest):
    index = get_index()
    data = {
        "base_price": req.base_price,
        "competitor_price": req.competitor_price or req.base_price * 1.05,
        "category": req.category,
    }
    features = engineer_features(data)
    results = index.search(features, k=req.k)
    return {"query": data, "similar_products": results}


@app.get("/metrics")
def metrics():
    m = load_metrics()
    if _recent_X:
        recent_demands = [float(predict(_model, x.reshape(1, -1))[0]) for x in _recent_X[-100:]]
        health = prediction_health_check(recent_demands)
    else:
        health = {"status": "no_data"}
    return {"model_metrics": m, "prediction_health": health, "n_recent_predictions": len(_recent_X)}


@app.get("/predictions")
def recent_predictions(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "product_id": r.product_id,
            "category": r.category,
            "predicted_demand": r.predicted_demand,
            "optimized_price": r.optimized_price,
            "created_at": str(r.created_at),
        }
        for r in rows
    ]
