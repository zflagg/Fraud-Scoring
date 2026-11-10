import pandas as pd
import numpy as np

import preprocessing
from config import NUM_COLS, CAT_COLS, TARGET

# Load data, split by time

df = pd.read_csv("train_transaction.csv")
df["TransactionAmt"] = np.log1p(df["TransactionAmt"]) # Tames to skew of specific values


train, val, test = preprocessing.time_split(df)



# Model 1: Logistic Regression
# Simplest model and provides a quick fit

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def as_str(d):
    d = d.copy(); d[CAT_COLS] = d[CAT_COLS].astype(str); return d

prep = make_column_transformer(
    (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CAT_COLS),
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_COLS)
)
logreg = make_pipeline(prep, LogisticRegression(class_weight="balanced", max_iter=1000)) # balanced tells the model that missing one is worse than a false alarm

logreg.fit(as_str(train)[CAT_COLS + NUM_COLS], train[TARGET])
scores = logreg.predict_proba(as_str(test)[CAT_COLS + NUM_COLS])[:,1]
print(preprocessing.evaluate(test[TARGET], scores))


# Model 2: XGBoost
# Gradient boosted trees, model to beat, handles missing values natively and no need to scale

from xgboost import XGBClassifier
# need categoreies to be learned from train only, then applied to val/test
train_x, val_x, test_x = train.copy(), val.copy(), test.copy()
for c in CAT_COLS:
    cats = train_x[c].astype("category").cat.categories
    for d in (train_x, val_x, test_x):
        d[c] = pd.Categorical(d[c], categories=cats)

xgb = XGBClassifier(
    n_estimators=2000,          # maximum number of trees; early stopping picks the real number
    learning_rate=0.05,         # how much each tree corrects; smaller is slower but usually better
    max_depth=8,                # how complex each tree can be
    subsample=0.8,              # each tree sees 80% of rows, which reduces overfitting
    colsample_bytree=0.5,       # and 50% of columns
    tree_method="hist", enable_categorical=True,
    eval_metric="aucpr",        # watch PR-AUC on the validation set
    early_stopping_rounds=100,  # stop when validation stops improving
)
X = CAT_COLS + NUM_COLS
xgb.fit(train_x[X], train_x[TARGET], eval_set=[(val_x[X], val_x[TARGET])], verbose=100)
print(preprocessing.evaluate(test[TARGET], xgb.predict_proba(test_x[X])[:, 1]))


# Model 3: NN with embeddings
# Each email domain or card becomes a learned vector, so the model can learn what domains behave simlarly
# Exports cleanly to ONNX

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

pre = preprocessing.Preprocessor().fit(train)

def to_tensors(d):
    x_cat, x_num = pre.transform(d)
    return (torch.tensor(x_cat, dtype=torch.long),
            torch.tensor(x_num, dtype=torch.float32),
            torch.tensor(d[TARGET].values, dtype=torch.float32)
            )

train_dl = DataLoader(TensorDataset(*to_tensors(train)), batch_size=1024, shuffle=True) # Shuffling is fine since it was already split by time
val_cat, val_num, val_y = to_tensors(val)

class FraudNet(nn.Module):
    def __init__(self, cardinalities, n_numeric):
        super().__init__()
        # One embedding table per categorical column, small columns get small vectors
        self.embs = nn.ModuleList(
            nn.Embedding(n + 1, min(16, (n + 1) // 2 + 1)) for n in cardinalities)
        emb_total = sum(e.embedding_dim for e in self.embs)
        self.mlp = nn.Sequential(
            nn.Linear(emb_total + n_numeric, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))       # one output, a raw fraud score

    def forward(self, x_cat, x_num):
        looked_up = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(looked_up + [x_num], dim=1)   # glue embeddings and numbers into one long row
        return self.mlp(x).squeeze(1)

model = FraudNet([len(pre.cat_maps[c]) for c in CAT_COLS], len(NUM_COLS))

# Training loop

y = train[TARGET].values
pos_weight = torch.tensor(np.sqrt((y==0).sum() / (y==1).sum()), dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

best, patience = 0, 0

for epoch in range(30):
    model.train()
    for x_cat, x_num, yb in train_dl:
        opt.zero_grad()
        loss = loss_fn(model(x_cat, x_num), yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(val_cat, val_num)).numpy()
    m = preprocessing.evaluate(val_y.numpy(), val_scores)
    print(epoch, m)

    if m["pr_auc"] > best:
        best, patience = m["pr_auc"], 0
        torch.save(model.state_dict(), "best.pt")
    else:
        patience += 1
        if patience == 3:
            break

# Reload best weights and score the test set once, log everything to MLFlow

import joblib
import mlflow
import mlflow.pytorch

# Rebuild the model and load the best weights
best_model = FraudNet([len(pre.cat_maps[c]) for c in CAT_COLS], len(NUM_COLS))
best_model.load_state_dict(torch.load("best.pt"))
best_model.eval()                         # dropout off, batchnorm uses its saved statistics

# Score the test set
test_cat, test_num, test_y = to_tensors(test)     # same Preprocessor, fit on train only
with torch.no_grad():                             # no gradients needed, which is faster and lighter
    test_scores = torch.sigmoid(best_model(test_cat, test_num)).numpy()

test_metrics = preprocessing.evaluate(test_y.numpy(), test_scores)
print("TEST:", test_metrics)

# --- 3. Log to MLflow ---
mlflow.set_experiment("fraud-scoring")
with mlflow.start_run(run_name="fraudnet-mlp-emb"):
    mlflow.log_params({
        "model": "FraudNet", "hidden": "256-64", "dropout": 0.3,
        "lr": 1e-3, "batch_size": 1024, "pos_weight": float(pos_weight),
        "n_cat": len(CAT_COLS), "n_num": len(NUM_COLS),
    })
    mlflow.log_metric("best_val_pr_auc", best)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    joblib.dump(pre, "preprocessor.pkl")
    mlflow.log_artifact("preprocessor.pkl")       # the encoders travel with the model
    mlflow.pytorch.log_model(best_model, "model")