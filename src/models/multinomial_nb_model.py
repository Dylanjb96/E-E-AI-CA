from sklearn.naive_bayes import MultinomialNB

class MultinomialNBModel(MultinomialNB):
    def __init__(self):
        super().__init__()