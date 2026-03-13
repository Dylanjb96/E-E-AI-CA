from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.models.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model_t2 = RandomForestClassifier(random_state=42)
        self.model_t23 = RandomForestClassifier(random_state=42)
        self.model_t234 = RandomForestClassifier(random_state=42)

    def train(self, data_bundle):
        self.model_t2.fit(data_bundle.X_train, data_bundle.y_train_t2)
        self.model_t23.fit(data_bundle.X_train, data_bundle.y_train_t23)
        self.model_t234.fit(data_bundle.X_train, data_bundle.y_train_t234)

    def predict(self, data_bundle):
        return {
            "t2": self.model_t2.predict(data_bundle.X_test),
            "t23": self.model_t23.predict(data_bundle.X_test),
            "t234": self.model_t234.predict(data_bundle.X_test),
        }

    def print_results(self, data_bundle, predictions):
        acc_t2 = accuracy_score(data_bundle.y_test_t2, predictions["t2"])
        acc_t23 = accuracy_score(data_bundle.y_test_t23, predictions["t23"])
        acc_t234 = accuracy_score(data_bundle.y_test_t234, predictions["t234"])

        print("=== Results ===")
        print(f"Type 2 accuracy: {acc_t2:.4f}")
        print(f"Type 2 + Type 3 accuracy: {acc_t23:.4f}")
        print(f"Type 2 + Type 3 + Type 4 accuracy: {acc_t234:.4f}")