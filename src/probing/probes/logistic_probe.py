from sklearn.linear_model import LogisticRegression
from .base_probe import BaseProbe

class LogisticProbe(BaseProbe):

    def __init__(self, seed=0):
        super().__init__(seed)
        self.model = LogisticRegression(random_state=self.seed)

    def fit(self, x_tr, y_tr):
        self.model.fit(x_tr, y_tr)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]

    def score(self, x, y):
        return self.model.score(x, y)