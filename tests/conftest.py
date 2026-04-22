import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TEST_DB_URL = "sqlite:///./test_price_prophet.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    from app.database import Base
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    import os
    if os.path.exists("./test_price_prophet.db"):
        os.remove("./test_price_prophet.db")


@pytest.fixture
def db(setup_db):
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def client(setup_db):
    import os
    os.environ["DATABASE_URL"] = TEST_DB_URL
    from app.database import get_db
    from app.main import app

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
