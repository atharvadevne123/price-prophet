import json
import logging
import os
from typing import Dict, List

import numpy as np
from scipy.stats import ks_2samp

from app.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

REFERENCE_STATS_PATH = os.getenv("REFERENCE_STATS_PATH", "reference_stats.json")


def compute_drift(reference: List[float], current: List[float]) -> Dict:
    if len(reference) < 5 or len(current) < 5:
        return {"error": "Not enough data for drift test", "drift_detected": False}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def compute_feature_drift(
    reference_matrix: np.ndarray, current_matrix: np.ndarray
) -> Dict:
    results = {}
    n_features = min(reference_matrix.shape[1], current_matrix.shape[1], len(FEATURE_NAMES))
    for i in range(n_features):
        fname = FEATURE_NAMES[i]
        results[fname] = compute_drift(
            reference_matrix[:, i].tolist(),
            current_matrix[:, i].tolist(),
        )
    drifted = [k for k, v in results.items() if v.get("drift_detected")]
    return {
        "feature_results": results,
        "drifted_features": drifted,
        "drift_detected": len(drifted) > 0,
        "drift_rate": round(len(drifted) / max(n_features, 1), 3),
    }


def save_reference_stats(X: np.ndarray) -> None:
    stats = {
        "mean": X.mean(axis=0).tolist(),
        "std": X.std(axis=0).tolist(),
        "min": X.min(axis=0).tolist(),
        "max": X.max(axis=0).tolist(),
        "n_samples": X.shape[0],
        "feature_names": FEATURE_NAMES,
    }
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Reference stats saved: n_samples=%d", X.shape[0])


def load_reference_stats() -> Dict:
    if os.path.exists(REFERENCE_STATS_PATH):
        with open(REFERENCE_STATS_PATH) as f:
            return json.load(f)
    return {}


def prediction_health_check(predictions: List[float]) -> Dict:
    if not predictions:
        return {"status": "no_data"}
    arr = np.array(predictions)
    return {
        "count": len(predictions),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std()), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "negative_pct": round(float((arr < 0).mean()) * 100, 2),
        "zero_pct": round(float((arr == 0).mean()) * 100, 2),
    }
