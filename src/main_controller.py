from src.preprocessing.data_loader import load_dataset, split_dataset
from src.features.vectorizer import vectorize_text
from src.data_models.dataset_bundle import DatasetBundle
from src.models.random_forest_model import RandomForestModel

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

    model = RandomForestModel()
    model.train(data_bundle)
    predictions = model.predict(data_bundle)
    model.print_results(data_bundle, predictions)

if __name__ == "__main__":
    main()