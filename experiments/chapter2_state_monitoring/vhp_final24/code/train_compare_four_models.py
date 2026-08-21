from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from v14_compute_control import DEFAULT_CPU_THREADS, configure_compute


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "02_final_cache"
OUT = ROOT / "03_four_model_comparison"
WINDOW = 24
SEED = 20260812


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


class SmallANN(nn.Module):
    """Synchronous baseline with no target history and no explicit bottleneck objective."""

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(16, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallGRU(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_inputs, 32, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Dropout(0.05), nn.Linear(32, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, width: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.depthwise = nn.Conv1d(width, width, kernel, dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = nn.functional.pad(x, (self.left_pad, 0))
        z = self.depthwise(z)
        z = nn.functional.gelu(z)
        z = self.pointwise(z)
        z = self.norm(z)
        return residual + self.dropout(z)


class SmallTCN(nn.Module):
    """Causal 29-minute receptive field; a 24-point window uses all available history."""

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        width = 16
        self.proj = nn.Conv1d(n_inputs, width, 1)
        self.blocks = nn.Sequential(
            CausalDepthwiseBlock(width, kernel=5, dilation=1),
            CausalDepthwiseBlock(width, kernel=5, dilation=2),
            CausalDepthwiseBlock(width, kernel=5, dilation=4),
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x.transpose(1, 2))
        z = self.blocks(z)
        return self.head(z[:, :, -1])


class SmallITransformer(nn.Module):
    """Compact variable-token iTransformer-style encoder for current-output estimation."""

    def __init__(self, n_inputs: int, n_outputs: int, window: int) -> None:
        super().__init__()
        d_model = 32
        self.n_outputs = n_outputs
        self.history_projection = nn.Linear(window, d_model)
        self.input_token_embedding = nn.Parameter(torch.zeros(1, n_inputs, d_model))
        self.output_queries = nn.Parameter(torch.empty(1, n_outputs, d_model))
        nn.init.normal_(self.output_queries, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=64,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, norm=nn.LayerNorm(d_model))
        self.readout = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.history_projection(x.transpose(1, 2)) + self.input_token_embedding
        queries = self.output_queries.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([tokens, queries], dim=1))
        return self.readout(encoded[:, -self.n_outputs :, :]).squeeze(-1)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    window: int
    batch_size: int
    max_epochs: int
    factory: Callable[[int, int], nn.Module]
    architecture: str


def specs() -> dict[str, ModelSpec]:
    return {
        "ANN": ModelSpec(
            "ANN",
            1,
            8192,
            400,
            lambda ni, no: SmallANN(ni, no),
            "17→32→16→21；LayerNorm+SiLU+Dropout(0.05)，当前输入同步软测量",
        ),
        "GRU-RNN": ModelSpec(
            "GRU-RNN",
            WINDOW,
            4096,
            300,
            lambda ni, no: SmallGRU(ni, no),
            "24×17→单层GRU(32)→LayerNorm→21；仅使用过去24分钟输入",
        ),
        "TCN": ModelSpec(
            "TCN",
            WINDOW,
            4096,
            300,
            lambda ni, no: SmallTCN(ni, no),
            "24×17→1×1投影(16)→3个因果深度可分离残差块(k=5,d=1/2/4)→21",
        ),
        "iTransformer-small": ModelSpec(
            "iTransformer-small",
            WINDOW,
            2048,
            300,
            lambda ni, no: SmallITransformer(ni, no, WINDOW),
            "24点历史投影为17个变量token；1层4头Encoder(d=32,ff=64)+21输出查询token",
        ),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def valid_window_ends(time_s: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return np.arange(len(time_s), dtype=np.int64)
    deltas = np.diff(time_s)
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        return np.empty(0, dtype=np.int64)
    # The regular production cache is one minute.  Physics prototypes may use
    # deterministic five-minute subsampling; infer the nominal cadence while
    # still rejecting every missing/duplicate edge in a sequence window.
    expected_step = int(np.median(positive))
    edge_bad = deltas != expected_step
    prefix = np.r_[0, np.cumsum(edge_bad, dtype=np.int64)]
    ends = np.arange(window - 1, len(time_s), dtype=np.int64)
    bad_count = prefix[ends] - prefix[ends - window + 1]
    return ends[bad_count == 0]


def make_samples(x: np.ndarray, y: np.ndarray, time_s: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ends = valid_window_ends(time_s, window)
    if window == 1:
        return x[ends].astype(np.float32, copy=False), y[ends].astype(np.float32, copy=False), time_s[ends]
    view = np.lib.stride_tricks.sliding_window_view(x, (window, x.shape[1])).squeeze(1)
    starts = ends - window + 1
    return view[starts].astype(np.float32, copy=False), y[ends].astype(np.float32, copy=False), time_s[ends]


def loader(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def model_loss(model: nn.Module, data: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for xb, yb in data:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        loss = torch.mean((model(xb) - yb) ** 2)
        total += float(loss) * len(xb)
        count += len(xb)
    return total / max(count, 1)


def selection_train(
    spec: ModelSpec,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]], int, float, float]:
    seed_everything(seed)
    model = spec.factory(fit_x.shape[-1], fit_y.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    train_loader = loader(fit_x, fit_y, spec.batch_size, True)
    val_loader = loader(val_x, val_y, spec.batch_size * 2, False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    min_epochs, patience = 45, 30
    started = time.perf_counter()
    for epoch in range(1, spec.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        lr_used = float(optimizer.param_groups[0]["lr"])
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = torch.mean((model(xb) - yb) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        train_loss = total / max(count, 1)
        val_loss = model_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_used})
        scheduler.step(val_loss)
        if val_loss < best_loss * (1 - 1e-4):
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss, time.perf_counter() - started


def full_train(
    spec: ModelSpec,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    seed: int,
    best_epoch: int,
    selection_history: list[dict[str, float]],
) -> tuple[nn.Module, list[dict[str, float]], float]:
    seed_everything(seed)
    model = spec.factory(x.shape[-1], y.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    data = loader(x, y, spec.batch_size, True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        lr = selection_history[min(epoch - 1, len(selection_history) - 1)]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total, count = 0.0, 0
        for xb, yb in data:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = torch.mean((model(xb) - yb) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        history.append({"epoch": epoch, "full_train_loss": total / max(count, 1), "lr": lr})
    return model, history, time.perf_counter() - started


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, batch: int, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    dummy = np.zeros((len(x), 1), dtype=np.float32)
    data = loader(x, dummy, batch * 2, False)
    parts = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for xb, _ in data:
        parts.append(model(xb.to(device, non_blocking=True)).float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(parts), time.perf_counter() - started


def point_metrics(
    protocol: str,
    algorithm: str,
    split: str,
    truth: np.ndarray,
    pred: np.ndarray,
    target_tags: list[str],
    target_meta: dict[str, dict[str, str]],
    train_truth: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, tag in enumerate(target_tags):
        train_range = float(np.ptp(train_truth[:, i]))
        rmse = math.sqrt(mean_squared_error(truth[:, i], pred[:, i]))
        mae = mean_absolute_error(truth[:, i], pred[:, i])
        rows.append(
            {
                "protocol": protocol,
                "algorithm": algorithm,
                "split": split,
                "target_tag": tag,
                "name_cn": target_meta[tag]["name_cn"],
                "unit": target_meta[tag]["unit"],
                "group": target_meta[tag]["group"],
                "r2": r2_score(truth[:, i], pred[:, i]),
                "rmse": rmse,
                "mae": mae,
                "nrmse_train_range": rmse / train_range if train_range > 0 else np.nan,
                "nmae_train_range": mae / train_range if train_range > 0 else np.nan,
                "train_actual_min": float(train_truth[:, i].min()),
                "train_actual_max": float(train_truth[:, i].max()),
                "actual_min": float(truth[:, i].min()),
                "actual_max": float(truth[:, i].max()),
                "pred_min": float(pred[:, i].min()),
                "pred_max": float(pred[:, i].max()),
                "samples": len(truth),
            }
        )
    return pd.DataFrame(rows)


def loss_plot(protocol: str, spec: ModelSpec, selection: pd.DataFrame, full: pd.DataFrame) -> None:
    configure_chinese_font()
    dest = OUT / "loss_curves"
    dest.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(selection.epoch, selection.train_loss, label="训练段Loss", color="#4C72B0")
    axes[0].plot(selection.epoch, selection.val_loss, label="验证段Loss", color="#C44E52")
    best_idx = int(selection.val_loss.idxmin())
    axes[0].axvline(selection.loc[best_idx, "epoch"], color="#55A868", ls="--", lw=1, label="最佳epoch")
    axes[0].set_yscale("log")
    axes[0].set_title("训练期尾段验证与早停")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("标准化MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(full.epoch, full.full_train_loss, color="#8172B2", label="完整训练月Loss")
    axes[1].set_yscale("log")
    axes[1].set_title("选定轮数后用完整训练月重训")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("标准化MSE")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{protocol} | {spec.name} | Loss曲线")
    fig.tight_layout()
    fig.savefig(dest / f"{protocol}_{spec.name}_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_one(protocol: str, spec: ModelSpec, metadata: dict[str, object], device: torch.device) -> tuple[pd.DataFrame, dict[str, object]]:
    data = np.load(CACHE / f"{protocol}.npz")
    raw_train_x = data["train_x"].astype(np.float64)
    raw_test_x = data["test_x"].astype(np.float64)
    raw_train_y = data["train_y"].astype(np.float64)
    raw_test_y = data["test_y"].astype(np.float64)
    train_time = data["train_time"]
    test_time = data["test_time"]
    input_tags = list(metadata["baseline_inputs"])
    target_tags = list(metadata["targets"])
    target_meta = metadata["target_metadata"]

    # A point may be all-but-constant in a particular training month.  It is
    # excluded before scaling, preventing unseen test variation from becoming
    # dozens of standard deviations.  No target point is removed.
    input_std = raw_train_x.std(axis=0)
    live_mask = np.isfinite(input_std) & (input_std > 1e-6)
    live_inputs = [tag for tag, live in zip(input_tags, live_mask) if live]
    dropped_inputs = [tag for tag, live in zip(input_tags, live_mask) if not live]
    raw_train_x = raw_train_x[:, live_mask]
    raw_test_x = raw_test_x[:, live_mask]

    split = int(len(raw_train_x) * 0.8)
    select_x_scaler = StandardScaler().fit(raw_train_x[:split])
    select_y_scaler = StandardScaler().fit(raw_train_y[:split])
    fit_x_raw = select_x_scaler.transform(raw_train_x[:split]).astype(np.float32)
    val_x_raw = select_x_scaler.transform(raw_train_x[split:]).astype(np.float32)
    fit_y_raw = select_y_scaler.transform(raw_train_y[:split]).astype(np.float32)
    val_y_raw = select_y_scaler.transform(raw_train_y[split:]).astype(np.float32)
    fit_x, fit_y, _ = make_samples(fit_x_raw, fit_y_raw, train_time[:split], spec.window)
    val_x, val_y, _ = make_samples(val_x_raw, val_y_raw, train_time[split:], spec.window)

    selection_model, selection_history, best_epoch, best_val_loss, selection_seconds = selection_train(
        spec, fit_x, fit_y, val_x, val_y, device, SEED
    )
    del selection_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    x_scaler = StandardScaler().fit(raw_train_x)
    y_scaler = StandardScaler().fit(raw_train_y)
    train_x_raw = x_scaler.transform(raw_train_x).astype(np.float32)
    test_x_raw = x_scaler.transform(raw_test_x).astype(np.float32)
    train_y_scaled_raw = y_scaler.transform(raw_train_y).astype(np.float32)
    test_y_scaled_raw = y_scaler.transform(raw_test_y).astype(np.float32)
    train_x, train_y_scaled, train_time_samples = make_samples(train_x_raw, train_y_scaled_raw, train_time, spec.window)
    test_x, test_y_scaled, test_time_samples = make_samples(test_x_raw, test_y_scaled_raw, test_time, spec.window)
    full_model, full_history, full_seconds = full_train(
        spec, train_x, train_y_scaled, device, SEED, best_epoch, selection_history
    )
    pred_train_scaled, infer_train_seconds = predict(full_model, train_x, spec.batch_size, device)
    pred_test_scaled, infer_test_seconds = predict(full_model, test_x, spec.batch_size, device)
    pred_train = y_scaler.inverse_transform(pred_train_scaled)
    pred_test = y_scaler.inverse_transform(pred_test_scaled)
    actual_train = y_scaler.inverse_transform(train_y_scaled)
    actual_test = y_scaler.inverse_transform(test_y_scaled)

    model_dir = OUT / "models" / protocol / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": full_model.state_dict(),
            "algorithm": spec.name,
            "architecture": spec.architecture,
            "window": spec.window,
            "live_inputs": live_inputs,
            "target_tags": target_tags,
            "input_scaler_mean": x_scaler.mean_,
            "input_scaler_scale": x_scaler.scale_,
            "target_scaler_mean": y_scaler.mean_,
            "target_scaler_scale": y_scaler.scale_,
        },
        model_dir / "model.pt",
    )
    selection_df = pd.DataFrame(selection_history)
    full_df = pd.DataFrame(full_history)
    selection_df.to_csv(model_dir / "selection_loss.csv", index=False, encoding="utf-8-sig")
    full_df.to_csv(model_dir / "full_train_loss.csv", index=False, encoding="utf-8-sig")
    (model_dir / "configuration.json").write_text(
        json.dumps(
            {
                "algorithm": spec.name,
                "architecture": spec.architecture,
                "parameter_count": count_parameters(full_model),
                "window": spec.window,
                "live_inputs": live_inputs,
                "dropped_near_constant_inputs": dropped_inputs,
                "targets": target_tags,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8-sig",
    )
    prediction_dir = OUT / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_dir / f"{protocol}_{spec.name}.npz",
        train_time=train_time_samples,
        train_actual=actual_train.astype(np.float32),
        train_pred=pred_train.astype(np.float32),
        test_time=test_time_samples,
        test_actual=actual_test.astype(np.float32),
        test_pred=pred_test.astype(np.float32),
        target_tags=np.asarray(target_tags),
    )
    loss_plot(protocol, spec, selection_df, full_df)

    metric_train = point_metrics(protocol, spec.name, "train", actual_train, pred_train, target_tags, target_meta, actual_train)
    metric_test = point_metrics(protocol, spec.name, "test", actual_test, pred_test, target_tags, target_meta, actual_train)
    metrics = pd.concat([metric_train, metric_test], ignore_index=True)
    test_rows = metrics[metrics.split == "test"]
    train_rows = metrics[metrics.split == "train"]
    summary = {
        "protocol": protocol,
        "algorithm": spec.name,
        "architecture": spec.architecture,
        "window": spec.window,
        "parameter_count": count_parameters(full_model),
        "live_input_count": len(live_inputs),
        "dropped_near_constant_inputs": dropped_inputs,
        "best_epoch": best_epoch,
        "epochs_run": len(selection_history),
        "best_val_loss": best_val_loss,
        "selection_seconds": selection_seconds,
        "full_train_seconds": full_seconds,
        "inference_test_seconds": infer_test_seconds,
        "train_samples": len(actual_train),
        "test_samples": len(actual_test),
        "macro_train_r2": float(train_rows.r2.mean()),
        "macro_test_r2": float(test_rows.r2.mean()),
        "macro_r2_gap": float(train_rows.r2.mean() - test_rows.r2.mean()),
        "macro_test_rmse": float(test_rows.rmse.mean()),
        "macro_test_mae": float(test_rows.mae.mean()),
        "macro_test_nrmse_train_range": float(test_rows.nrmse_train_range.mean()),
        "macro_test_nmae_train_range": float(test_rows.nmae_train_range.mean()),
        "device": str(device),
        "seed": SEED,
    }
    return metrics, summary


def comparison_plots(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    configure_chinese_font()
    for protocol in summary.protocol.unique():
        sub = summary[summary.protocol == protocol].sort_values("macro_test_r2", ascending=False)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.bar(sub.algorithm, sub.macro_test_r2, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(sub)])
        ax.set_ylim(max(0, sub.macro_test_r2.min() - 0.08), min(1.005, sub.macro_test_r2.max() + 0.03))
        ax.set_ylabel("测试集宏平均 R²")
        ax.set_title(f"{protocol} | 四算法测试集宏平均R²")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, sub.macro_test_r2):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(OUT / f"{protocol}_algorithm_macro_r2.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        test = metrics[(metrics.protocol == protocol) & (metrics.split == "test")]
        pivot = test.pivot(index="name_cn", columns="algorithm", values="r2")
        order = list(sub.algorithm)
        pivot = pivot.reindex(columns=order)
        fig, ax = plt.subplots(figsize=(9.5, max(7.5, len(pivot) * 0.38)))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=max(-0.2, np.nanmin(pivot.to_numpy())), vmax=1.0)
        ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        ax.set_title(f"{protocol} | 各输出点测试R²")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(image, ax=ax, label="R²")
        fig.tight_layout()
        fig.savefig(OUT / f"{protocol}_per_point_r2_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", nargs="*", default=["month_2024", "month_2025"])
    parser.add_argument("--algorithms", nargs="*", default=list(specs()))
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    args = parser.parse_args()
    configure_compute(args.cpu_threads)
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8-sig"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; torch={torch.__version__}; gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)
    all_metrics: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    selected_specs = specs()
    for protocol in args.protocols:
        for algorithm in args.algorithms:
            print(f"START {protocol} | {algorithm}", flush=True)
            metrics, summary = run_one(protocol, selected_specs[algorithm], metadata, device)
            all_metrics.append(metrics)
            summaries.append(summary)
            print(
                f"DONE {protocol} | {algorithm} | test R2={summary['macro_test_r2']:.6f} | "
                f"gap={summary['macro_r2_gap']:.6f} | epoch={summary['best_epoch']}",
                flush=True,
            )
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    metrics_df.to_csv(OUT / "metrics_by_point_all_algorithms.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "algorithm_comparison.csv", index=False, encoding="utf-8-sig")
    group = (
        metrics_df[metrics_df.split == "test"]
        .groupby(["protocol", "algorithm", "group"], as_index=False)
        .agg(macro_r2=("r2", "mean"), macro_rmse=("rmse", "mean"), macro_mae=("mae", "mean"), outputs=("target_tag", "count"))
    )
    group.to_csv(OUT / "algorithm_comparison_by_output_group.csv", index=False, encoding="utf-8-sig")
    comparison_plots(summary_df, metrics_df)
    manifest = {
        "task": metadata["task"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "seed": SEED,
        "algorithms": [asdict(selected_specs[name]) | {"factory": str(selected_specs[name].factory)} for name in args.algorithms],
        "protocols": metadata["protocols"],
        "baseline_inputs": metadata["baseline_inputs"],
        "targets": metadata["targets"],
        "no_target_history": True,
        "scaler_fit_scope": "training data only; validation selection scaler fits first 80% of training month",
        "window_gap_rule": "all adjacent timestamps in a sequence must be exactly 60 seconds",
        "summaries": summaries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8-sig")
    print(summary_df[["protocol", "algorithm", "parameter_count", "macro_train_r2", "macro_test_r2", "macro_r2_gap"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
