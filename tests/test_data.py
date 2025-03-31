import pytest
from src.data_loader import load_data

def test_data_loading():
    """Test que verifica que los datos se cargan correctamente"""
    X, y = load_data()
    assert X.shape[0] == y.shape[0], "X e y deben tener el mismo número de muestras"
    assert 'Amount' in X.columns, "Columna 'Amount' debe estar presente"
    assert set(y.unique()).issubset({0, 1}), "Target debe ser binario (0, 1)"