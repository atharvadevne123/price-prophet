
from app.features import (
    FEATURE_NAMES,
    engineer_batch_features,
    engineer_features,
    generate_synthetic_training_data,
)


def test_feature_vector_length():
    features = engineer_features({"base_price": 100.0, "category": "electronics"})
    assert len(features) == len(FEATURE_NAMES)


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 15


def test_weekend_flag():
    # 2024-01-06 is a Saturday
    feat_sat = engineer_features({"base_price": 50.0, "date": "2024-01-06"})
    feat_mon = engineer_features({"base_price": 50.0, "date": "2024-01-08"})
    assert feat_sat[5] == 1.0  # is_weekend
    assert feat_mon[5] == 0.0


def test_holiday_season():
    feat_dec = engineer_features({"base_price": 50.0, "date": "2024-12-15"})
    feat_jun = engineer_features({"base_price": 50.0, "date": "2024-06-15"})
    assert feat_dec[6] == 1.0  # is_holiday_season
    assert feat_jun[6] == 0.0


def test_price_ratio_computed():
    feat = engineer_features({"base_price": 100.0, "competitor_price": 200.0})
    assert abs(feat[2] - 0.5) < 0.01  # price_ratio = 100/200


def test_batch_features_shape():
    records = [
        {"base_price": 100.0, "category": "electronics"},
        {"base_price": 200.0, "category": "clothing"},
        {"base_price": 50.0, "category": "food"},
    ]
    X = engineer_batch_features(records)
    assert X.shape == (3, len(FEATURE_NAMES))


def test_synthetic_data_generation():
    X, y = generate_synthetic_training_data(n_samples=100)
    assert X.shape == (100, len(FEATURE_NAMES))
    assert y.shape == (100,)
    assert (y >= 0).all()


def test_category_encoding():
    feat_elec = engineer_features({"base_price": 100.0, "category": "electronics"})
    feat_unknown = engineer_features({"base_price": 100.0, "category": "unknown_cat"})
    assert feat_elec[7] == 0  # electronics = 0
    assert feat_unknown[7] == 8  # fallback = 8


def test_demand_trend_positive():
    feat = engineer_features({
        "base_price": 100.0,
        "historical_demand_7d": 100,
        "historical_demand_30d": 200,
    })
    # (100*4 - 200) / 200 = 1.0
    assert feat[12] > 0
