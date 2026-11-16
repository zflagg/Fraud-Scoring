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