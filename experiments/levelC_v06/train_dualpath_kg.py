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
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--adjacency', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
args = parser.parse_args()

torch.set_num_threads(1)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

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

class DualPathKGGCN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.register_buffer('Ahat', torch.tensor(Ahat))
        self.local = nn.Linear(L, hidden)
        self.graph = nn.Linear(L, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden * 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N * hidden * 2, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 4),
        )

    def forward(self, x):
        h_local = torch.relu(self.local(x))
        h_graph = torch.relu(self.graph(torch.matmul(self.Ahat, x)))
        h = self.norm(torch.cat([h_local, h_graph], dim=-1))
        return self.classifier(h)

model = DualPathKGGCN(hidden=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=128, shuffle=True)
xv, yvv = torch.tensor(Xv), torch.tensor(yv)

best_state = None
best_f1 = -1.0
best_loss = float('inf')
bad = 0
for epoch in range(1, 61):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(xv)
        val_loss = loss_fn(logits, yvv).item()
        val_pred = logits.argmax(1).numpy()
        val_f1 = f1_score(yv, val_pred, average='macro')

    if val_f1 > best_f1 + 1e-4 or (abs(val_f1 - best_f1) <= 1e-4 and val_loss < best_loss):
        best_f1, best_loss, best_epoch = val_f1, val_loss, epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= 8:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred = model(torch.tensor(Xte)).argmax(1).numpy()

result = pd.DataFrame([{
    'seed': seed,
    'accuracy': accuracy_score(yte, pred),
    'macro_precision': precision_score(yte, pred, average='macro', zero_division=0),
    'macro_recall': recall_score(yte, pred, average='macro', zero_division=0),
    'macro_f1': f1_score(yte, pred, average='macro'),
    'best_val_macro_f1': best_f1,
    'best_epoch': best_epoch,
}])

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False, encoding='utf-8-sig')
print(result.to_string(index=False))
