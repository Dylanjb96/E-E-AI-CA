from src.config import MODEL_NAME
from src.models.chained_multioutput_model import ChainedMultiOutputModel
from src.models.random_forest_model import RandomForestModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.multinomial_nb_model import MultinomialNBModel

def get_model():
    if MODEL_NAME == "random_forest":
        return ChainedMultiOutputModel(RandomForestModel)
    elif MODEL_NAME == "logistic_regression":
        return ChainedMultiOutputModel(LogisticRegressionModel)
    elif MODEL_NAME == "multinomial_nb":
        return ChainedMultiOutputModel(MultinomialNBModel)
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")