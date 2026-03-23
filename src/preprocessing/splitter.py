from sklearn.model_selection import train_test_split

from src.config import (
    TEXT_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)

def split_dataset(df):
    X = df[TEXT_COLUMN]
    y_t2 = df["target_t2"]
    y_t23 = df["target_t23"]
    y_t234 = df["target_t234"]

    return train_test_split(
        X,
        y_t2,
        y_t23,
        y_t234,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_t2,
    )