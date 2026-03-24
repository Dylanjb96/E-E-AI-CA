from src.preprocessing.data_loader import load_dataset
from src.preprocessing.splitter import split_dataset

def test_dataset_loads_and_splits():
    df = load_dataset()
    assert not df.empty
    assert "target_t2" in df.columns
    assert "target_t23" in df.columns
    assert "target_t234" in df.columns

    result = split_dataset(df)
    assert len(result) == 8

    X_train, X_test, y_train_t2, y_test_t2, y_train_t23, y_test_t23, y_train_t234, y_test_t234 = result
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train_t2) > 0
    assert len(y_test_t234) > 0