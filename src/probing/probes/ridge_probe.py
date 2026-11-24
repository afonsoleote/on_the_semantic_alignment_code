import numpy as np

from sklearn.linear_model import RidgeClassifier
from .base_probe import BaseProbe
import warnings


class RidgeProbe(BaseProbe):

    def __init__(self, seed, alpha=1.0):
        super().__init__(seed)
        self.alpha = alpha
        self.model = RidgeClassifier(alpha=alpha)

        warnings.filterwarnings(
            "ignore",
            message="Singular matrix in solving dual problem.*"
        )
    def fit(self, x_tr, y_tr, x_val=None, y_val=None, alpha_grid=None):

        if alpha_grid is None:
            self.model.fit(x_tr, y_tr)
            return

        best_acc = -1

        for alpha in alpha_grid:
            model = RidgeClassifier(alpha=alpha)
            model.fit(x_tr, y_tr)

            acc = model.score(x_val, y_val)

            if acc > best_acc:
                best_acc = acc
                self.alpha = alpha
                self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        # scores = self.model.decision_function(x)
        # return 1 / (1 + np.exp(-scores))
        return self.model.predict(x)