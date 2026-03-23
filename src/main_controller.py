from src.config import MODEL_NAME
from src.preprocessing.data_loader import load_dataset
from src.preprocessing.splitter import split_dataset
from src.features.vectorizer import vectorize_text
from src.data_models.dataset_bundle import DatasetBundle
from src.models.model_factory import get_model
from src.evaluation.metrics import evaluate_predictions, print_evaluation

def main():
    df = load_dataset()

    (
        X_train,
        X_test,
        y_train_t2,
        y_test_t2,
        y_train_t23,
        y_test_t23,
        y_train_t234,
        y_test_t234,
    ) = split_dataset(df)

    X_train_vec, X_test_vec, _ = vectorize_text(X_train, X_test)

    data_bundle = DatasetBundle(
        X_train=X_train_vec,
        X_test=X_test_vec,
        y_train_t2=y_train_t2,
        y_test_t2=y_test_t2,
        y_train_t23=y_train_t23,
        y_test_t23=y_test_t23,
        y_train_t234=y_train_t234,
        y_test_t234=y_test_t234,
    )

    model = get_model()
    model.train(data_bundle)
    predictions = model.predict(data_bundle)
    results = evaluate_predictions(data_bundle, predictions)
    print("--- " + MODEL_NAME + " ---")
    print_evaluation(results)

if __name__ == "__main__":
    main()