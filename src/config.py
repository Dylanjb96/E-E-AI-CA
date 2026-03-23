from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASET_NAME = "Purchasing.csv"
DATA_PATH = DATA_DIR / DATASET_NAME

MODEL_NAME = "random_forest"

TEXT_COLUMN = "Interaction content"
TYPE2_COLUMN = "Type 2"
TYPE3_COLUMN = "Type 3"
TYPE4_COLUMN = "Type 4"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_CLASS_COUNT = 2
MAX_FEATURES = 5000