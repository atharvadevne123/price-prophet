import os
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./price_prophet.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, index=True)
    category = Column(String)
    predicted_demand = Column(Float)
    optimized_price = Column(Float)
    confidence = Column(Float)
    features_used = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True)
    auc_mean = Column(Float)
    auc_std = Column(Float)
    rmse = Column(Float)
    n_features = Column(Integer)
    n_samples = Column(Integer)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String)
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)  # 0 or 1
    logged_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
