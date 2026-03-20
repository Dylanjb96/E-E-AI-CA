from sklearn.naive_bayes import MultinomialNB
from src.models.base_model import BaseModel

class MultinomialNBModel(BaseModel):
    def __init__(self):
        self.model_t2 = MultinomialNB()
        self.model_t23 = MultinomialNB()
        self.model_t234 = MultinomialNB()

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
