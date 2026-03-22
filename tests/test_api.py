import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавление пути к src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.app import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_missing_model():
    response = client.post("/predict", json={
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    })

    assert response.status_code in [200, 503]

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_type" in response.json()