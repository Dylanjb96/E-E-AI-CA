from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(LogisticRegression):
    def __init__(self):
        super().__init__(max_iter=1000, random_state=42)