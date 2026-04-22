import numpy as np
import pandas as pd
from typing import Dict, Any


FEATURE_NAMES = [
    "base_price",
    "competitor_price",
    "price_ratio",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday_season",
    "category_encoded",
    "stock_level",
    "days_since_last_promotion",
    "historical_demand_7d",
    "historical_demand_30d",
    "demand_trend",
    "price_elasticity_estimate",
    "margin_ratio",
]


CATEGORY_MAP = {
    "electronics": 0,
    "clothing": 1,
    "food": 2,
    "home": 3,
    "sports": 4,
    "beauty": 5,
    "toys": 6,
    "books": 7,
    "other": 8,
}

HOLIDAY_MONTHS = {11, 12, 1}


def engineer_features(data: Dict[str, Any]) -> np.ndarray:
    base_price = float(data.get("base_price") or 100.0)
    competitor_price = float(data.get("competitor_price") or base_price * 1.05)
    price_ratio = base_price / max(competitor_price, 0.01)

    date_str = data.get("date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    dt = pd.Timestamp(date_str)
    day_of_week = dt.dayofweek
    month = dt.month
    is_weekend = int(day_of_week >= 5)
    is_holiday_season = int(month in HOLIDAY_MONTHS)

    category = data.get("category", "other").lower()
    category_encoded = CATEGORY_MAP.get(category, 8)

    stock_level = float(data.get("stock_level") or 100)
    days_since_promo = float(data.get("days_since_last_promotion") or 30)
    hist_demand_7d = float(data.get("historical_demand_7d") or 50)
    hist_demand_30d = float(data.get("historical_demand_30d") or 200)

    demand_trend = (hist_demand_7d * 4 - hist_demand_30d) / max(hist_demand_30d, 1)

    cost = float(data.get("cost") or base_price * 0.6)
    margin_ratio = (base_price - cost) / max(base_price, 0.01)

    price_elasticity_estimate = -1.5 * (1 + 0.5 * (price_ratio - 1))

    features = np.array([
        base_price,
        competitor_price,
        price_ratio,
        day_of_week,
        month,
        is_weekend,
        is_holiday_season,
        category_encoded,
        stock_level,
        days_since_promo,
        hist_demand_7d,
        hist_demand_30d,
        demand_trend,
        price_elasticity_estimate,
        margin_ratio,
    ], dtype=np.float32)

    return features


def engineer_batch_features(records: list) -> np.ndarray:
    return np.vstack([engineer_features(r) for r in records])


def generate_synthetic_training_data(n_samples: int = 2000) -> tuple:
    np.random.seed(42)
    records = []
    labels = []

    categories = list(CATEGORY_MAP.keys())

    for _ in range(n_samples):
        base_price = np.random.uniform(10, 500)
        competitor_price = base_price * np.random.uniform(0.8, 1.3)
        cost = base_price * np.random.uniform(0.4, 0.7)
        stock = np.random.randint(0, 500)
        hist_7d = np.random.uniform(10, 200)
        hist_30d = hist_7d * np.random.uniform(3.5, 5.0)
        days_promo = np.random.randint(0, 90)
        category = np.random.choice(categories)

        dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(np.random.randint(0, 365)))

        record = {
            "base_price": base_price,
            "competitor_price": competitor_price,
            "cost": cost,
            "stock_level": stock,
            "historical_demand_7d": hist_7d,
            "historical_demand_30d": hist_30d,
            "days_since_last_promotion": days_promo,
            "category": category,
            "date": dt.strftime("%Y-%m-%d"),
        }
        records.append(record)

        price_ratio = base_price / max(competitor_price, 0.01)
        demand = (
            hist_7d * 0.4
            + (1 / price_ratio) * 20
            + (1 if dt.month in HOLIDAY_MONTHS else 0) * 15
            + np.random.normal(0, 5)
        )
        labels.append(max(demand, 0))

    X = engineer_batch_features(records)
    y = np.array(labels, dtype=np.float32)
    return X, y
