from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASET_NAME = "AppGallery.csv" #AppGallery.csv" #Purchasing.csv" (Switch between datasets here by removing the '#' from the desired dataset and adding it to the other)
DATA_PATH = DATA_DIR / DATASET_NAME

MODEL_NAME =  "random_forest"  # "logistic_regression" #"multinomial_nb" (Switch between models here by removing the '#' from the desired model and adding it to the others)

TEXT_COLUMN = "Interaction content"
TYPE2_COLUMN = "Type 2"
TYPE3_COLUMN = "Type 3"
TYPE4_COLUMN = "Type 4"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_CLASS_COUNT = 2
MAX_FEATURES = 5000