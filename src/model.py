from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

class FraudDetector:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state
        )
    
    def train(self, X_train, y_train):
        """Entrena el modelo"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo"""
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        return {
            'report': report,
            'roc_auc': roc_auc
        }