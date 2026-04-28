

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_forecast_basic(client):
    payload = {
        "product_id": "TEST-001",
        "base_price": 99.99,
        "competitor_price": 109.99,
        "category": "electronics",
        "stock_level": 200,
        "historical_demand_7d": 75,
        "historical_demand_30d": 300,
    }
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["product_id"] == "TEST-001"
    assert data["predicted_demand"] >= 0
    assert "optimized_price" in data
    assert data["recommendation"] in ("increase", "decrease", "hold")


def test_forecast_missing_price(client):
    resp = client.post("/forecast", json={"product_id": "BAD"})
    assert resp.status_code == 422


def test_train_endpoint(client):
    resp = client.post("/train", json={"n_samples": 200})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "trained"
    assert "run_id" in data
    assert data["rmse_mean"] >= 0


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_metrics" in data
    assert "n_recent_predictions" in data


def test_similar_products(client):
    resp = client.post("/similar", json={"base_price": 150.0, "category": "electronics", "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "similar_products" in data
    assert len(data["similar_products"]) <= 3


def test_predictions_history(client):
    client.post("/forecast", json={
        "product_id": "HIST-001",
        "base_price": 50.0,
        "category": "clothing",
    })
    resp = client.get("/predictions?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_drift_insufficient_data(client):
    resp = client.get("/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "n_recent" in data or "drift_detected" in data


def test_forecast_different_categories(client):
    categories = ["electronics", "clothing", "food", "home", "sports"]
    for cat in categories:
        resp = client.post("/forecast", json={
            "product_id": f"CAT-{cat}",
            "base_price": 80.0,
            "category": cat,
        })
        assert resp.status_code == 200, f"Failed for category: {cat}"


def test_forecast_price_boundaries(client):
    for price in [0.01, 10.0, 1000.0, 9999.0]:
        resp = client.post("/forecast", json={
            "product_id": "PRICE-TEST",
            "base_price": price,
            "category": "other",
        })
        assert resp.status_code == 200
        assert resp.json()["predicted_demand"] >= 0


def test_categories_endpoint(client):
    resp = client.get("/categories")
    assert resp.status_code == 200
    data = resp.json()
    assert "categories" in data
    assert "count" in data
    assert data["count"] > 0
    assert "electronics" in data["categories"]
    assert "clothing" in data["categories"]
    assert sorted(data["categories"]) == data["categories"]


def test_summary_endpoint(client):
    resp = client.get("/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "total_training_runs" in data
    assert "drift_events_logged" in data
    assert "n_recent_in_memory" in data
    assert data["total_predictions"] >= 0


def test_predictions_limit_validation(client):
    resp = client.get("/predictions?limit=0")
    assert resp.status_code == 422

    resp = client.get("/predictions?limit=201")
    assert resp.status_code == 422

    resp = client.get("/predictions?limit=50")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_forecast_with_cost_field(client):
    resp = client.post("/forecast", json={
        "product_id": "COST-001",
        "base_price": 100.0,
        "cost": 40.0,
        "category": "electronics",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "expected_profit" in data
    assert data["optimized_price"] > 0


def test_train_sample_bounds(client):
    resp = client.post("/train", json={"n_samples": 99})
    assert resp.status_code == 422

    resp = client.post("/train", json={"n_samples": 50001})
    assert resp.status_code == 422
