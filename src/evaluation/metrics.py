from sklearn.metrics import accuracy_score

def evaluate_predictions(data_bundle, predictions):
    return {
        "t2_accuracy": accuracy_score(data_bundle.y_test_t2, predictions["t2"]),
        "t23_accuracy": accuracy_score(data_bundle.y_test_t23, predictions["t23"]),
        "t234_accuracy": accuracy_score(data_bundle.y_test_t234, predictions["t234"]),
    }

def print_evaluation(results):
    print("=== Chained Multi-Output Results ===")
    print(f"Type 2 accuracy: {results['t2_accuracy']:.4f}")
    print(f"Type 2 + Type 3 accuracy: {results['t23_accuracy']:.4f}")
    print(f"Type 2 + Type 3 + Type 4 accuracy: {results['t234_accuracy']:.4f}")