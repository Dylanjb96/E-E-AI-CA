from src.preprocessing.data_loader import load_dataset

def test_chained_targets_are_created():
    df = load_dataset()

    assert "target_t2" in df.columns
    assert "target_t23" in df.columns
    assert "target_t234" in df.columns

    first_row = df.iloc[0]

    assert first_row["target_t2"] == first_row["Type 2"]
    assert " || " in first_row["target_t23"]
    assert " || " in first_row["target_t234"]