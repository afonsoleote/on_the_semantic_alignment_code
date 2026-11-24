from abc import ABC, abstractmethod

class BaseProbe(ABC):

    def __init__(self, seed=None):
        self.seed = seed

    @abstractmethod
    def fit(self, x_tr, y_tr):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def predict_proba(self, x):
        raise NotImplementedError