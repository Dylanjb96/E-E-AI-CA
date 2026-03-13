from src.preprocessing.data_loader import load_dataset, split_dataset

def test_dataset_loads_and_splits():
    df = load_dataset()
    assert not df.empty
    assert "target_t2" in df.columns
    assert "target_t23" in df.columns
    assert "target_t234" in df.columns

    result = split_dataset(df)
    assert len(result) == 8