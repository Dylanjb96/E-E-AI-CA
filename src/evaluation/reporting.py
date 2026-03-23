
def print_evaluation(results):
    print("=== Chained Multi-Output Results ===")
    print(f"Type 2 accuracy: {results['t2_accuracy']:.4f}")
    print(f"Type 2 + Type 3 accuracy: {results['t23_accuracy']:.4f}")
    print(f"Type 2 + Type 3 + Type 4 accuracy: {results['t234_accuracy']:.4f}")