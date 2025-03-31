import pandas as pd

def load_data(path='data/creditcard.csv'):
    """Carga los datos de fraude"""
    df = pd.read_csv(path)
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    return X, y

def get_train_test_split(test_size=0.3, random_state=42):
    """Obtiene split de entrenamiento y prueba"""
    X, y = load_data()
    return train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
