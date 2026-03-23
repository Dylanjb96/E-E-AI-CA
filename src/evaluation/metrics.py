from sklearn.metrics import accuracy_score

def evaluate_predictions(data_bundle, predictions):
    return {
        "t2_accuracy": accuracy_score(data_bundle.y_test_t2, predictions["t2"]),
        "t23_accuracy": accuracy_score(data_bundle.y_test_t23, predictions["t23"]),
        "t234_accuracy": accuracy_score(data_bundle.y_test_t234, predictions["t234"]),
    }
