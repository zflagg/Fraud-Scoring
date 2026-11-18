import pandas as pd
import numpy as np
import sys

import preprocessing
from config import NUM_COLS, CAT_COLS, TARGET

# FUNC: log
# DESC: Forced prints, flush=true forces the print to my current understanding. Also makes it easier to add a timestamp or something
def log(msg):
    print(msg, flush=True)

# Load data, split by time

def load_data(n_rows=None):
    log("Loading data...")
    df = pd.read_csv("train_transaction.csv")
    df["TransactionAmt"] = np.log1p(df["TransactionAmt"]) # Tames to skew of specific values
    if n_rows:
        df = df.sort_values("TransactionDT").head(n_rows)
    # Returns train, val, test
    return preprocessing.time_split(df)



# Model 1: Logistic Regression
# Simplest model and provides a quick fit
def run_logreg(train, val, test):
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
    print("Logistic Regression: ")
    print(preprocessing.evaluate(test[TARGET], scores))


# Model 2: XGBoost
# Gradient boosted trees, model to beat, handles missing values natively and no need to scale

def run_xgboost(train, val, test):
    from xgboost import XGBClassifier
    # need categoreies to be learned from train only, then applied to val/test
    train_x, val_x, test_x = train.copy(), val.copy(), test.copy()
    for c in CAT_COLS:
        for d in (train_x, val_x, test_x):
            d[c] = d[c].astype("string")          # 111.0 becomes "111.0"; missing stays missing
        cats = train_x[c].astype("category").cat.categories
        for d in (train_x, val_x, test_x):
            d[c] = pd.Categorical(d[c], categories=cats)
    log("Boosting...")
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

######### Moved largely to FNModel File #############
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from FNModelmodel import FraudNet


if __name__ == "__main__":
    which = sys.argv[1]
    train, val, test = load_data(100,000)
    {"logreg": run_logreg, "xgboost": run_xgboost, "fraudnet": run_fraudnet}[which](train, val, test)