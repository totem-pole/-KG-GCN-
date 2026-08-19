"""Train one Level-C graph model seed at a time.

This script is intentionally timeout-safe: one invocation trains exactly one adjacency/seed
and writes one CSV. Multi-seed aggregation is handled by summarize_results.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def row_norm(A: np.ndarray) -> np.ndarray:
    deg = A.sum(1)
    deg = np.where(deg > 0, deg, 1.0)
    return (A / deg[:, None]).astype(np.float32)


class OrderedSensorGCN(nn.Module):
    """Self path + graph-neighbor path + node-aware readout.

    The node order is fixed by the 24 DCS tags; we do not apply global mean pooling.
    This preserves the engineering identity of each measurement point.
    """

    def __init__(self, A: np.ndarray, window: int = 60, hidden: int = 8, n_classes: int = 4):
        super().__init__()
        n_nodes = A.shape[0]
        self.n_nodes = n_nodes
        self.register_buffer("A", torch.tensor(row_norm(A)))
        self.self_fc = nn.Linear(window, hidden)
        self.neigh_fc = nn.Linear(window, hidden, bias=False)
        self.gate = nn.Parameter(torch.tensor(-1.5))  # conservative initial neighbor weight
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_nodes * hidden, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self_term = self.self_fc(x)
        neigh_x = torch.matmul(self.A, x)
        h = self_term + torch.sigmoid(self.gate) * self.neigh_fc(neigh_x)
        return self.classifier(h)


def train_one(dataset: str, adjacency: str, seed: int, out_csv: str, out_cm: str | None = None) -> None:
    torch.set_num_threads(1)
    seed_all(seed)

    d = np.load(dataset, allow_pickle=True)
    Xtr, ytr = d["X_train"].astype(np.float32), d["y_train"].astype(np.int64)
    Xv, yv = d["X_val"].astype(np.float32), d["y_val"].astype(np.int64)
    Xte, yte = d["X_test"].astype(np.float32), d["y_test"].astype(np.int64)
    tags = list(d["tags"].astype(str))

    A_df = pd.read_csv(adjacency, index_col=0)
    A_df = A_df.loc[tags, tags]
    A = A_df.to_numpy(np.float32)

    model = OrderedSensorGCN(A, window=Xtr.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=128,
        shuffle=True,
    )
    xv, yvv = torch.tensor(Xv), torch.tensor(yv)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad = 0
    patience = 10

    for epoch in range(1, 71):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yvv).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).cpu().numpy()
        gate = float(torch.sigmoid(model.gate).detach().cpu())

    result = pd.DataFrame([{
        "seed": seed,
        "accuracy": accuracy_score(yte, pred),
        "macro_precision": precision_score(yte, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(yte, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "learned_neighbor_gate": gate,
        "graph_edges": int(A.sum() / 2),
    }])

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))

    if out_cm:
        labels = ["F0正常", "F1通流侵蚀", "F2一级抽汽支路泄漏", "F3VHP汽封泄漏"]
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3])
        Path(out_cm).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_cm, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--adjacency", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cm", default=None)
    args = ap.parse_args()
    train_one(args.dataset, args.adjacency, args.seed, args.out, args.cm)


if __name__ == "__main__":
    main()
