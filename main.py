import pandas as pd
import numpy as np

# Auto log things to mlflow
import mlflow
mlflow.autolog()

# FUNC: Time Split
# DESC: It's crucial to keep track of time in this setting, since a random split will leak future behavior to the test set

def time_split(df, time_col="TransactionDT", val_frac=0.15, test_frac=0.15):
    df = df.sort_values(time_col)
    n = len(df)
    a, b = int(n * (1 - val_frac - test_frac), int(n * (1 - test_frac)))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


# FUNC: Recall and Evaluate
# DESC: instead of evaluating ourselves on the high volume of true negatives, we will look at smaller volume of true and false positives

from sklearn.metrics import average_precision_score, roc_curve

def recall_at_fpr(y_true, y_score, max_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return tpr[fpr <= max_fpr].max()

def evaluate(y_true, y_score):
    return {"pr_auc": average_precision_score(y_true, y_score),
            "recall@1%fpr": recall_at_fpr(y_true, y_score)}



# Preprocessing

class Preprocessor:
    def fit(self, df):
        self.cat_maps = {c: {v: i + 1 for i, v in enumerate(df[c].dropna().unique())}
                         for c in CAT_COLS}                      
        self.means = df[NUM_COLS].mean()
        self.stds = df[NUM_COLS].std() + 1e-6
        return self

    def transform(self, df):
        x_cat = np.stack([df[c].map(self.cat_maps[c]).fillna(0).astype(int) for c in CAT_COLS], axis=1)
        x_num = ((df[NUM_COLS] - self.means) / self.stds).fillna(0).to_numpy()
        return x_cat, x_num

# Save it in MLFlow

with mlflow.start_run():
    pre = Preprocessor().fit(train)
    # ... train model on pre.transform(train) ...
    joblib.dump(pre, "preprocessor.pkl")
    mlflow.log_artifact("preprocessor.pkl")
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_metrics(evaluate(y_test, scores))

# Create the torch model

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class FraudNet(nn.Module):
    def __init__(self, cardinalities, n_numeric, emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c + 1, emb_dim) for c in cardinalities)  # +1 for unseen bucket
        self.mlp = nn.Sequential(
            nn.Linear(len(cardinalities) * emb_dim + n_numeric, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x_cat, x_num):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [x_num], dim=1)).squeeze(1)



# FUNC: Velocity Update
# DESC: We can't scan a DB how frequently a card has been used in the last hour, so we have it precomputed (Redis)
def velocity_update(state: dict, txn_time: int, amount: float, window_s: int = 3600) -> dict:
    events = [(t, a) for t, a in state.get("events", []) if txn_time - t < window_s]
    events.append((txn_time, amount))
    return {"events": events, "count_1h": len(events), "sum_1h": sum(a for _, a in events)}