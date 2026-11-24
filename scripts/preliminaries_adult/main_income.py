import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.utils.common_paths import DATASETS_PATH
from src.utils.models.adult_mlp import AdultMLP

DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

CSV =  DATASETS_PATH / "adult" / 'adult_clean.csv'
CODE_JSON = DATASETS_PATH / "adult" / 'adult_encodings.json'
CKPT_PATH = DATASETS_PATH / "adult" /'adult_income_mlp.pth'
PREP_JSON = 'adult_prep.json'

MAX_EPOCHS = 200
PATIENCE = 20
BATCH = 256
LR = 1e-3

TARGET = 'income'

class AdultPreprocessor:
    CONTINUOUS = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    NOMINAL = ['workclass', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'native-country']

    def __init__(self, encodings, scaler=None):
        self.encodings = encodings
        self.scaler = scaler
        self.nominal = [c for c in self.NOMINAL]

    def fit(self, df):
        cont = df[self.CONTINUOUS].astype(float).values
        self.scaler = {'mean': cont.mean(axis=0).tolist(),
                       'std': (cont.std(axis=0) + 1e-8).tolist()}
        return self

    def transform(self, df):
        cont = df[self.CONTINUOUS].astype(float).values
        cont = (cont - np.array(self.scaler['mean'])) / np.array(self.scaler['std'])
        onehots = []
        for col in self.nominal:
            n = len(self.encodings[col])
            oh = np.zeros((len(df), n), dtype=float)
            oh[np.arange(len(df)), df[col].values.astype(int)] = 1.0
            onehots.append(oh)
        return np.concatenate([cont] + onehots, axis=1).astype('float32')

    @property
    def in_dim(self):
        return len(self.CONTINUOUS) + sum(len(self.encodings[c]) for c in self.nominal)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'encodings': self.encodings, 'scaler': self.scaler,
                       'nominal': self.nominal}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)
        obj = cls(state['encodings'], state['scaler'])
        obj.nominal = state['nominal']
        return obj

if __name__ == "__main__":
    with open(CODE_JSON) as f:
        encodings = json.load(f)['encodings']

    df = pd.read_csv(CSV)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    prep = AdultPreprocessor(encodings).fit(train_df)
    x_train = prep.transform(train_df)
    x_test = prep.transform(test_df)
    y_train = train_df[TARGET].values.astype('float32').reshape(-1, 1)
    y_test = test_df[TARGET].values.astype('float32').reshape(-1, 1)
    print(train_df)
    print(train_df.columns)
    print(len(train_df.columns))
    print(x_train.shape)
    print(f'[INFO] device = {DEVICE}')
    print(f'[INFO] input dim = {prep.in_dim} (features only, concepts dropped)')
    print(f'[INFO] train {len(x_train)}, test {len(x_test)}')
    print(f'[INFO] positive rate train = {y_train.mean():.4f}')

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_train))
    n_val = int(0.15 * len(x_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train[tr_idx]), torch.tensor(y_train[tr_idx])),
        batch_size=BATCH, shuffle=True)
    valid_loader = DataLoader(
        TensorDataset(torch.tensor(x_train[val_idx]), torch.tensor(y_train[val_idx])),
        batch_size=512)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test), torch.tensor(y_test)),
        batch_size=512)

    model = AdultMLP.build(prep.in_dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), LR)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    wait = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                preds = (model(xb.to(DEVICE)) > 0).cpu()
                correct += (preds == yb.bool()).sum().item()
                total += yb.size(0)
        accuracy = correct / total
        print(f'[INFO] epoch {epoch}: val acc = {accuracy:.4f}')

        if accuracy > best_acc:
            best_acc = accuracy
            wait = 0
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f'[INFO] early stop at epoch {epoch}, best {best_acc:.4f}')
                break

    prep.save(PREP_JSON)

    print('[INFO] loading best checkpoint for test evaluation')
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    true_positive = 0
    true_negative = 0
    positives = 0
    negatives = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = (model(xb.to(DEVICE)) > 0).cpu()
            yb = yb.bool()
            true_positive += (preds & yb).sum().item()
            true_negative += (~preds & ~yb).sum().item()
            positives += yb.sum().item()
            negatives += (~yb).sum().item()

    sensitivity = true_positive / positives
    specificity = true_negative / negatives
    balanced_accuracy = (sensitivity + specificity) / 2
    plain_accuracy = (true_positive + true_negative) / (positives + negatives)
    print(f'[INFO] test plain acc    = {plain_accuracy:.4f}')
    print(f'[INFO] test balanced acc = {balanced_accuracy:.4f}  '
          f'(sens {sensitivity:.4f}, spec {specificity:.4f})')