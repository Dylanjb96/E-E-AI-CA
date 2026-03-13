from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, data_bundle):
        pass

    @abstractmethod
    def predict(self, data_bundle):
        pass

    @abstractmethod
    def print_results(self, data_bundle, predictions):
        pass