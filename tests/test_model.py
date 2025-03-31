import pytest
import numpy as np
from src.model import FraudDetector
from src.data_loader import load_data

@pytest.fixture
def data():
    X, y = load_data()
    return X[:100], y[:100]  # Usamos solo 100 muestras para pruebas

def test_model_training(data):
    """Test del entrenamiento del modelo"""
    X, y = data
    model = FraudDetector()
    model.train(X, y)
    assert hasattr(model.model, 'feature_importances_'), "Modelo no entrenado correctamente"

def test_model_prediction(data):
    """Test de las predicciones del modelo"""
    X, y = data
    model = FraudDetector()
    model.train(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y), "Número de predicciones incorrecto"
    assert set(preds).issubset({0, 1}), "Predicciones deben ser 0 o 1"