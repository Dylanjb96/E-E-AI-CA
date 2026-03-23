import pandas as pd
from src.preprocessing.text_cleaner import clean_text

from src.config import (
    DATA_PATH,
    TEXT_COLUMN,
    TYPE2_COLUMN,
    TYPE3_COLUMN,
    TYPE4_COLUMN,
    MIN_CLASS_COUNT,
)

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    df = df[[TEXT_COLUMN, TYPE2_COLUMN, TYPE3_COLUMN, TYPE4_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

    df[TYPE2_COLUMN] = df[TYPE2_COLUMN].fillna("None")
    df[TYPE3_COLUMN] = df[TYPE3_COLUMN].fillna("None")
    df[TYPE4_COLUMN] = df[TYPE4_COLUMN].fillna("None")

    df = df[df[TEXT_COLUMN] != ""]

    # optional: remove rare Type 2 classes
    counts = df[TYPE2_COLUMN].value_counts()
    valid_classes = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df[TYPE2_COLUMN].isin(valid_classes)].copy()

    # chained targets for Design Choice 1
    df["target_t2"] = df[TYPE2_COLUMN]
    df["target_t23"] = df[TYPE2_COLUMN] + " || " + df[TYPE3_COLUMN]
    df["target_t234"] = (
        df[TYPE2_COLUMN] + " || " + df[TYPE3_COLUMN] + " || " + df[TYPE4_COLUMN]
    )

    return df
