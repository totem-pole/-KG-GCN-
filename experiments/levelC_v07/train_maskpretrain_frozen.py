from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--mask-ratio', type=float, default=0.15)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--adjacency', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

D = np.load(args.data, allow_pickle=True)
Xtr, ytr = D['X_train'], D['y_train']
Xv, yv = D['X_val'], D['y_val']
Xte, yte = D['X_test'], D['y_test']
A = pd.read_csv(args.adjacency, index_col=0).to_numpy(np.float32)
N, L = Xtr.shape[1], Xtr.shape[2]

At = A.copy()
np.fill_diagonal(At, 1.0)
deg = At.sum(1)
inv = 1 / np.sqrt(np.maximum(deg, 1e-8))
Ahat = (inv[:, None] * At * inv[None, :]).astype(np.float32)

class Encoder(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.register_buffer('Ahat', torch.tensor(Ahat))
        self.local = nn.Linear(L, hidden)
        self.graph = nn.Linear(L, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden * 2)
    def forward(self, x):
        h_local = torch.relu(self.local(x))
        h_graph = torch.relu(self.graph(torch.matmul(self.Ahat, x)))
        return self.norm(torch.cat([h_local, h_graph], dim=-1))

class MaskAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(32)
        self.decoder = nn.Linear(64, L)
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Stage 1: unlabeled masked reconstruction on TRAIN split only.
mask_ae = MaskAE()
optimizer = torch.optim.AdamW(mask_ae.parameters(), lr=1e-3, weight_decay=1e-4)
train_loader = DataLoader(TensorDataset(torch.tensor(Xtr)), batch_size=128, shuffle=True)
xv = torch.tensor(Xv)

best_state = None
best_val_mse = float('inf')
bad = 0
for epoch in range(1, 41):
    mask_ae.train()
    for (xb,) in train_loader:
        mask = torch.rand_like(xb) < args.mask_ratio
        masked_x = xb.masked_fill(mask, 0.0)
        optimizer.zero_grad()
        rec = mask_ae(masked_x)
        loss = ((rec - xb)[mask] ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mask_ae.parameters(), 5.0)
        optimizer.step()

    mask_ae.eval()
    with torch.no_grad():
        torch.manual_seed(5000 + epoch + args.seed)
        val_mask = torch.rand_like(xv) < args.mask_ratio
        val_rec = mask_ae(xv.masked_fill(val_mask, 0.0))
        val_mse = ((val_rec - xv)[val_mask] ** 2).mean().item()

    if val_mse < best_val_mse - 1e-4:
        best_val_mse = val_mse
        best_pretrain_epoch = epoch
        best_state = {k: v.detach().clone() for k, v in mask_ae.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= 6:
            break

mask_ae.load_state_dict(best_state)

# Stage 2: freeze encoder and train only FC+Softmax classifier.
encoder = mask_ae.encoder
for p in encoder.parameters():
    p.requires_grad = False

class FrozenClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N * 64, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 4),
        )
    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        return self.classifier(h)

model = FrozenClassifier(encoder)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1.5e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
cls_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=128, shuffle=True)
xvv, yvv = torch.tensor(Xv), torch.tensor(yv)

best_cls = None
best_f1 = -1.0
best_loss = float('inf')
bad = 0
for epoch in range(1, 61):
    model.train()
    for xb, yb in cls_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(xvv)
        val_loss = loss_fn(logits, yvv).item()
        val_pred = logits.argmax(1).numpy()
        val_f1 = f1_score(yv, val_pred, average='macro')

    if val_f1 > best_f1 + 1e-4 or (abs(val_f1 - best_f1) <= 1e-4 and val_loss < best_loss):
        best_f1, best_loss, best_cls_epoch = val_f1, val_loss, epoch
        best_cls = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= 8:
            break

model.load_state_dict(best_cls)
model.eval()
with torch.no_grad():
    pred = model(torch.tensor(Xte)).argmax(1).numpy()

result = pd.DataFrame([{
    'seed': args.seed,
    'mask_ratio': args.mask_ratio,
    'pretrain_best_epoch': best_pretrain_epoch,
    'pretrain_val_mse': best_val_mse,
    'classifier_best_epoch': best_cls_epoch,
    'best_val_macro_f1': best_f1,
    'accuracy': accuracy_score(yte, pred),
    'macro_precision': precision_score(yte, pred, average='macro', zero_division=0),
    'macro_recall': recall_score(yte, pred, average='macro', zero_division=0),
    'macro_f1': f1_score(yte, pred, average='macro'),
}])

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False, encoding='utf-8-sig')
print(result.to_string(index=False))
