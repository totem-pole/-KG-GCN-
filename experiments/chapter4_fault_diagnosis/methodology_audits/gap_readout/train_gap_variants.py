from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_subset(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 0.999:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cls in np.unique(y):
        ids = np.where(y == cls)[0]
        count = max(1, int(round(len(ids) * fraction)))
        chosen.extend(rng.choice(ids, count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def normalized_adjacency(path: str) -> np.ndarray:
    a = pd.read_csv(path, index_col=0).to_numpy(np.float32)
    np.fill_diagonal(a, 1.0)
    degree = a.sum(1)
    inv = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    return (inv[:, None] * a * inv[None, :]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adjacency", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["yang_gap", "dual_gap"], required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--classifier-hidden", type=int, default=96)
    ap.add_argument("--node-embed-dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--stage", choices=["select", "final"], default="select")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    torch.set_num_threads(1)
    seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = np.load(args.data, allow_pickle=True)
    x_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    x_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)
    x_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    n_nodes, window = x_train.shape[1], x_train.shape[2]
    ahat = normalized_adjacency(args.adjacency)

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("ahat", torch.tensor(ahat))
            self.input_map = nn.Linear(window, args.hidden)
            if args.node_embed_dim > 0:
                self.node_embedding = nn.Parameter(torch.empty(1, n_nodes, args.node_embed_dim))
                nn.init.normal_(self.node_embedding, mean=0.0, std=0.05)
                self.node_fusion = nn.Linear(args.hidden + args.node_embed_dim, args.hidden)
            else:
                self.node_embedding = None
                self.node_fusion = None
            self.gcn1 = nn.Linear(args.hidden, args.hidden, bias=False)
            self.gcn2 = nn.Linear(args.hidden, args.hidden, bias=False)
            self.norm1 = nn.LayerNorm(args.hidden)
            self.norm2 = nn.LayerNorm(args.hidden)
            self.dual_norm = nn.LayerNorm(args.hidden * 2)

        @property
        def output_dim(self) -> int:
            return args.hidden if args.mode == "yang_gap" else args.hidden * 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h0 = self.input_map(x)
            if self.node_embedding is not None:
                identity = self.node_embedding.expand(x.shape[0], -1, -1)
                h0 = self.node_fusion(torch.cat([h0, identity], dim=-1))
            h0 = torch.relu(h0)
            h1 = torch.relu(self.norm1(h0 + self.gcn1(torch.matmul(self.ahat, h0))))
            h2 = torch.relu(self.norm2(h1 + self.gcn2(torch.matmul(self.ahat, h1))))
            if args.mode == "yang_gap":
                return h2
            return self.dual_norm(torch.cat([h0, h2], dim=-1))

    class MaskAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.decoder = nn.Linear(self.encoder.output_dim, window)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    autoencoder = MaskAutoencoder().to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train)), batch_size=128, shuffle=True)
    x_val_tensor = torch.tensor(x_val, device=device)
    best_pretrain_state = None
    best_pretrain_mse = float("inf")
    best_pretrain_epoch = 0
    patience = 0
    for epoch in range(1, 41):
        autoencoder.train()
        for (xb_cpu,) in train_loader:
            xb = xb_cpu.to(device, non_blocking=True)
            mask = torch.rand_like(xb) < 0.15
            optimizer.zero_grad(set_to_none=True)
            reconstructed = autoencoder(xb.masked_fill(mask, 0.0))
            loss = ((reconstructed - xb)[mask] ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)
            optimizer.step()
        autoencoder.eval()
        with torch.no_grad():
            seed_all(args.seed + 7000 + epoch)
            val_mask = torch.rand_like(x_val_tensor) < 0.15
            val_rec = autoencoder(x_val_tensor.masked_fill(val_mask, 0.0))
            val_mse = ((val_rec - x_val_tensor)[val_mask] ** 2).mean().item()
        if val_mse < best_pretrain_mse - 1e-4:
            best_pretrain_mse = val_mse
            best_pretrain_epoch = epoch
            best_pretrain_state = {k: v.detach().cpu().clone() for k, v in autoencoder.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
    assert best_pretrain_state is not None
    autoencoder.load_state_dict(best_pretrain_state)
    autoencoder.to(device)
    encoder = autoencoder.encoder
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    class GAPClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Sequential(
                nn.Linear(encoder.output_dim, args.classifier_hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.classifier_hidden, 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                node_features = self.encoder(x)
            graph_features = node_features.mean(dim=1)  # strict global average pooling
            return self.classifier(graph_features)

    subset = stratified_subset(y_train, args.label_fraction, args.seed + 11000)
    model = GAPClassifier().to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1.5e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    cls_loader = DataLoader(
        TensorDataset(torch.tensor(x_train[subset]), torch.tensor(y_train[subset])),
        batch_size=128,
        shuffle=True,
    )
    xv = torch.tensor(x_val, device=device)
    yv = torch.tensor(y_val, device=device)
    best_cls_state = None
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_cls_epoch = 0
    patience = 0
    for epoch in range(1, 81):
        model.train()
        for xb_cpu, yb_cpu in cls_loader:
            xb, yb = xb_cpu.to(device, non_blocking=True), yb_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(xv)
            val_loss = loss_fn(logits, yv).item()
            val_pred = logits.argmax(1).cpu().numpy()
            val_f1 = f1_score(y_val, val_pred, average="macro")
        if val_f1 > best_val_f1 + 1e-4 or (abs(val_f1 - best_val_f1) <= 1e-4 and val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_cls_epoch = epoch
            best_cls_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    assert best_cls_state is not None
    model.load_state_dict(best_cls_state)
    model.to(device).eval()

    result = {
        "stage": args.stage,
        "model": args.mode,
        "seed": args.seed,
        "hidden": args.hidden,
        "classifier_hidden": args.classifier_hidden,
        "node_embed_dim": args.node_embed_dim,
        "dropout": args.dropout,
        "label_fraction": args.label_fraction,
        "device": str(device),
        "nodes": n_nodes,
        "window": window,
        "labeled_train_samples": int(len(subset)),
        "pretrain_epoch": best_pretrain_epoch,
        "pretrain_val_mse": best_pretrain_mse,
        "classifier_epoch": best_cls_epoch,
        "val_macro_f1": best_val_f1,
        "val_loss": best_val_loss,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stage}_{args.mode}_h{args.hidden}_e{args.node_embed_dim}_c{args.classifier_hidden}_d{args.dropout:g}_lf{args.label_fraction:g}_seed{args.seed}"
    if args.stage == "final":
        with torch.no_grad():
            test_pred = model(torch.tensor(x_test, device=device)).argmax(1).cpu().numpy()
        result.update(
            accuracy=accuracy_score(y_test, test_pred),
            macro_precision=precision_score(y_test, test_pred, average="macro", zero_division=0),
            macro_recall=recall_score(y_test, test_pred, average="macro", zero_division=0),
            macro_f1=f1_score(y_test, test_pred, average="macro"),
        )
        pd.DataFrame(
            confusion_matrix(y_test, test_pred),
            index=["F0", "F1", "F2", "F3"],
            columns=["F0", "F1", "F2", "F3"],
        ).to_csv(out / f"{stem}_confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(out / f"{stem}.csv", index=False, encoding="utf-8-sig")
    (out / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
