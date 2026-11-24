import lightgbm as lgb

from .base_probe import BaseProbe


class LGBMProbe(BaseProbe):

    def __init__(self, seed):
        super().__init__(seed)
        self.model = None

    def fit(self, x_tr, y_tr, x_val=None, y_val=None):

        tr_data = lgb.Dataset(x_tr, label=y_tr)
        val_data = lgb.Dataset(x_val, label=y_val)

        self.model = lgb.train(
            {
                "objective": "binary",
                "verbosity": -1,
                "seed": self.seed,
                "min_data_in_leaf": max(min(len(y_tr) // 10, 20), 1),
            },
            tr_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(15)],
        )

        return self

    def predict(self, x):
        return (self.model.predict(x) > 0.5).astype(int)

    def predict_proba(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        preds = self.predict(x)
        return (preds == y).mean()