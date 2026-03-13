from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import MAX_FEATURES

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer