import pandas as pd
import numpy as np
from config import NUM_COLS, CAT_COLS, TARGET

# Auto log things to mlflow
import mlflow
mlflow.autolog()


# FUNC: Time Split
# DESC: It's crucial to keep track of time in this setting, since a random split will leak future behavior to the test set

def time_split(df, time_col="TransactionDT", val_frac=0.15, test_frac=0.15):
    df = df.sort_values(time_col)
    n = len(df)
    a, b = int(n * (1 - val_frac - test_frac)), int(n * (1 - test_frac))
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
