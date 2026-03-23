from src.config import MODEL_NAME
from src.models.random_forest_model import RandomForestModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.multinomial_nb_model import MultinomialNBModel

def get_model():
    if MODEL_NAME == "random_forest":
        return RandomForestModel()
    elif MODEL_NAME == "logistic_regression":
        return LogisticRegressionModel()
    elif MODEL_NAME == "multinomial_nb":
        return MultinomialNBModel()
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")