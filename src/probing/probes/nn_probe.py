import copy

import torch
import torch.nn as nn

from .base_probe import BaseProbe


class NNProbe(BaseProbe):

    def __init__(self, architecture, weight_decay=0.0, loss_fn=None, seed=0):

        super().__init__(seed)
        torch.manual_seed(seed)

        self.architecture = architecture
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn

        self.model = None
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.optimizer = None

    def fit(self, x_tr, y_tr, x_val=None, y_val=None, epochs=150, batch_size=64, lr=1e-3, early_stopping=True, patience=15, verbose=False):

        torch.manual_seed(self.seed)

        self.model = self.architecture.build(x_tr.shape[1])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay,)

        x_tr = torch.as_tensor(x_tr, dtype=torch.float32)
        y_tr = torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1)

        x_val = torch.as_tensor(x_val, dtype=torch.float32)
        y_val = torch.as_tensor(y_val, dtype=torch.float32).view(-1, 1)

        n = x_tr.size(0)

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0
        tr_loss = 0
        val_loss = 0
        for epoch in range(epochs):
            self.model.train()

            perm = torch.randperm(n)

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]

                xb = x_tr[idx]
                yb = y_tr[idx]

                self.optimizer.zero_grad()

                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                tr_loss = loss
                loss.backward()
                self.optimizer.step()

            self.model.eval()

            with torch.no_grad():
                val_logits = self.model(x_val)
                val_loss = self.loss_fn(val_logits, y_val).item()

            if verbose:
                print(f"[INFO] Epoch {epoch:3d} | val_loss = {val_loss:.4f}")

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"[INFO] Early stopping at epoch {epoch}.")
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self, tr_loss, val_loss

    def predict_proba(self, x):

        self.model.eval()
        x = torch.tensor(x)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(-1)

        return probs.numpy()

    def predict(self, x):

        probs = self.predict_proba(x)
        return (probs > 0.5).astype(int)