from dataclasses import dataclass
from typing import Any

@dataclass
class DatasetBundle:
    X_train: Any
    X_test: Any
    y_train_t2: Any
    y_test_t2: Any
    y_train_t23: Any
    y_test_t23: Any
    y_train_t234: Any
    y_test_t234: Any