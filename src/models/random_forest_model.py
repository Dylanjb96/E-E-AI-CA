from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(RandomForestClassifier):
    def __init__(self):
        super().__init__(random_state=42)
