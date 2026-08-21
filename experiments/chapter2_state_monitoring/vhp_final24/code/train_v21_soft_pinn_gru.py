from __future__ import annotations

"""V21 same-architecture GRU versus equation-level soft PINN-GRU.

Every profile uses the identical predictor:

    24 x 17 -> one-layer GRU(32) -> LayerNorm -> Dropout(0.05) -> 21

All 21 outputs are direct neural predictions.  Physics appears only in the
training objective and is absent from inference; no physical solution replaces,
gates, blends, or overwrites a prediction.
"""

import argparse
import copy
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_compare_four_models as core
from audit_v21_measured_physics_residuals import (
    PIPE_EXT,
    PIPE_RH1,
    build_reference,
    equation_residuals,
)
from evaluate_v17_state_heads import (
    BLADE,
    EXT_P,
    EXT_T,
    HEATER_P,
    HEATER_T,
    POUT,
    VERTICAL_T,
    VHP_T,
)
from torch_if97_region2 import IF97Region2
from train_v18_end_to_end_pigru import FrozenBladeLogK, FrozenRidge, pipeline_spec
from v14_compute_control import configure_compute


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "21_v15_extreme_eta_pinn" / "cache"
OUT = ROOT / "43_v21_soft_pinn_gru"
MODELS = OUT / "models"
PREDICTIONS = OUT / "predictions"
FIGURES = OUT / "figures"
CALIBRATIONS = OUT / "calibration_cache"
for folder in (OUT, MODELS, PREDICTIONS, FIGURES, CALIBRATIONS):
    folder.mkdir(parents=True, exist_ok=True)

WINDOW = 24
PHYSICS_WINDOW = 26
DEFAULT_SEEDS = (20260813,)


@dataclass(frozen=True)
class Profile:
    name: str
    families: tuple[str, ...]
    adaptive: bool = False
    synthetic: bool = False
    description: str = ""


PROFILES: dict[str, Profile] = {
    "GRU": Profile("GRU", (), description="pure data baseline"),
    "PINN_STODOLA_FIXED": Profile(
        "PINN_STODOLA_FIXED",
        ("stodola",),
        description="cross-year accepted exhaust + extraction Stodola residuals",
    ),
    "PINN_STODOLA_ADAPTIVE": Profile(
        "PINN_STODOLA_ADAPTIVE",
        ("stodola",),
        adaptive=True,
        description="same audited Stodola residuals with gradient-norm/conflict-aware weight",
    ),
    "PINN_STODOLA_IF97_FIXED": Profile(
        "PINN_STODOLA_IF97_FIXED",
        ("stodola", "if97"),
        description="Stodola plus measured-state-audited IF97 efficiency consistency",
    ),
    "PINN_THERMO_ADAPTIVE": Profile(
        "PINN_THERMO_ADAPTIVE",
        ("stodola", "if97"),
        adaptive=True,
        description="audited Stodola + IF97 with gradient-conflict-aware weights",
    ),
    "PINN_THERMO_PIPE_ADAPTIVE": Profile(
        "PINN_THERMO_PIPE_ADAPTIVE",
        ("stodola", "if97", "pipe"),
        adaptive=True,
        description="thermodynamic constraints plus seven first-order pipe dynamics",
    ),
    "PINN_FULL_FIXED": Profile(
        "PINN_FULL_FIXED",
        ("stodola", "blade", "if97", "pipe"),
        description="all 21 points directly predicted; accepted/conditional physics only",
    ),
    "PINN_FULL_ADAPTIVE": Profile(
        "PINN_FULL_ADAPTIVE",
        ("stodola", "blade", "if97", "pipe"),
        adaptive=True,
        description="full soft PINN with gradient-norm/conflict-aware weights",
    ),
    "PINN_FULL_ADAPTIVE_SYNTH": Profile(
        "PINN_FULL_ADAPTIVE_SYNTH",
        ("stodola", "blade", "if97", "pipe"),
        adaptive=True,
        synthetic=True,
        description="adaptive full PINN plus unlabeled manifold-interpolated physics collocation",
    ),
}

BASE_WEIGHTS = {"stodola": 0.006, "blade": 0.003, "if97": 0.002, "pipe": 0.002}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def configure_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def split_index(time_s: np.ndarray) -> int:
    split_time = int(time_s[0]) + int(0.8 * (int(time_s[-1]) - int(time_s[0])))
    split = int(np.searchsorted(time_s, split_time, side="left"))
    return max(1000, min(len(time_s) - 500, split))


def make_samples(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    time_s: np.ndarray,
    xs: StandardScaler,
    ys: StandardScaler,
    window: int = PHYSICS_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = xs.transform(raw_x).astype(np.float32)
    y = ys.transform(raw_y).astype(np.float32)
    return core.make_samples(x, y, time_s, window)


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def robust_reference_spec(ref: dict[str, Any], inputs: list[str], targets: list[str]) -> dict[str, Any]:
    residuals = equation_residuals(ref["x"], ref["y"], ref["time"], inputs, targets, ref)

    def normalizer(name: str) -> dict[str, float]:
        item = residuals[name]
        return {"center": float(item["reference_center"]), "scale": float(item["reference_scale"])}

    pipe = {}
    for tag, model in ref["pipe_models"].items():
        pipe[tag] = {
            "a": float(model["a_per_minute"]),
            "bias": float(model["bias_c_per_minute"]),
            **normalizer(f"pipe_dynamics_{tag}"),
        }
    return {
        "history_label": ref["label"],
        "blade_cal": ref["blade_cal"],
        "exhaust_spec": pipeline_spec(ref["exhaust_model"], 6),
        "extraction_spec": pipeline_spec(ref["ext_models"]["compact"], 7),
        "eta_vhp_spec": pipeline_spec(ref["eta_models"]["vhp"], 2),
        "eta_vertical_spec": pipeline_spec(ref["eta_models"]["vertical"], 2),
        "eta_extraction_spec": pipeline_spec(ref["eta_models"]["extraction"], 2),
        "eta_heater_spec": pipeline_spec(ref["eta_models"]["heater"], 2),
        "normalizers": {
            "blade": normalizer("stodola_blade"),
            "exhaust": normalizer("stodola_exhaust"),
            "extraction": normalizer("stodola_extraction_compact"),
            "eta_vhp": normalizer("if97_eta_vhp"),
            "eta_vertical": normalizer("if97_eta_vertical"),
            "eta_extraction": normalizer("if97_eta_extraction"),
            "eta_heater": normalizer("if97_eta_heater"),
        },
        "pipe": pipe,
    }


def load_or_build_calibration(year: int, inputs: list[str], targets: list[str], rebuild: bool = False) -> dict[str, Any]:
    path = CALIBRATIONS / f"physics_calibration_{year}.pkl"
    if path.exists() and not rebuild:
        print(f"[physics] loading cached {year} prior", flush=True)
        with path.open("rb") as handle:
            return pickle.load(handle)
    print(f"[physics] building leakage-safe {year} prior", flush=True)
    calibration = robust_reference_spec(build_reference(year, inputs, targets), inputs, targets)
    with path.open("wb") as handle:
        pickle.dump(calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return calibration


class V21SoftPIGRU(nn.Module):
    def __init__(
        self,
        xs: StandardScaler,
        ys: StandardScaler,
        inputs: list[str],
        targets: list[str],
        calibration: dict[str, Any],
    ) -> None:
        super().__init__()
        # This is the only trainable prediction path in every profile.
        self.predictor = core.SmallGRU(len(inputs), len(targets))
        self.if97 = IF97Region2()
        self.blade_prior = FrozenBladeLogK(calibration["blade_cal"])
        self.exhaust_prior = FrozenRidge(calibration["exhaust_spec"])
        self.extraction_prior = FrozenRidge(calibration["extraction_spec"])
        self.eta_priors = nn.ModuleDict(
            {
                "vhp": FrozenRidge(calibration["eta_vhp_spec"]),
                "vertical": FrozenRidge(calibration["eta_vertical_spec"]),
                "extraction": FrozenRidge(calibration["eta_extraction_spec"]),
                "heater": FrozenRidge(calibration["eta_heater_spec"]),
            }
        )
        for name, value in (
            ("x_mean", xs.mean_),
            ("x_scale", xs.scale_),
            ("y_mean", ys.mean_),
            ("y_scale", ys.scale_),
        ):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.inputs, self.targets = inputs, targets
        self.p_idx = list(range(6))
        self.t_idx = list(range(6, 12))
        self.valve_idx = [inputs.index("SYG1_10FD1R-HFD1OM"), inputs.index("SYG1_10FD2R-HFD2OM")]
        self.load_idx = inputs.index("SYG1_MW")
        self.flow_idx = inputs.index("SYG1_10MAA50FP101-XQ01-OUT")
        self.ids = {
            "blade": [targets.index(t) for t in BLADE],
            "pout": [targets.index(t) for t in POUT],
            "vhp_t": [targets.index(t) for t in VHP_T],
            "vertical_t": [targets.index(t) for t in VERTICAL_T],
            "ext_p": [targets.index(t) for t in EXT_P],
            "ext_t": [targets.index(t) for t in EXT_T],
            "heater_p": [targets.index(t) for t in HEATER_P],
            "heater_t": [targets.index(t) for t in HEATER_T],
            "pipe_ext": [targets.index(t) for t in PIPE_EXT],
            "pipe_rh1": [targets.index(t) for t in PIPE_RH1],
        }
        self.normalizer_names: dict[str, tuple[str, str]] = {}
        for name, spec in calibration["normalizers"].items():
            center_name, scale_name = f"norm_{name}_center", f"norm_{name}_scale"
            self.register_buffer(center_name, torch.tensor(spec["center"], dtype=torch.float32))
            self.register_buffer(scale_name, torch.tensor(max(spec["scale"], 1e-6), dtype=torch.float32))
            self.normalizer_names[name] = (center_name, scale_name)
        self.pipe_specs: dict[int, tuple[str, str, str, str]] = {}
        for tag, spec in calibration["pipe"].items():
            safe = tag.replace("-", "_")
            names = tuple(f"pipe_{safe}_{suffix}" for suffix in ("a", "bias", "center", "scale"))
            for name, value in zip(names, (spec["a"], spec["bias"], spec["center"], max(spec["scale"], 1e-6))):
                self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
            self.pipe_specs[targets.index(tag)] = names

    def forward(self, x24: torch.Tensor) -> torch.Tensor:
        return self.predictor(x24)

    def raw_x(self, x24: torch.Tensor) -> torch.Tensor:
        return x24 * self.x_scale + self.x_mean

    def raw_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.y_scale + self.y_mean

    def normalise_residual(self, name: str, raw: torch.Tensor) -> torch.Tensor:
        center_name, scale_name = self.normalizer_names[name]
        return (raw - getattr(self, center_name)) / getattr(self, scale_name)

    @staticmethod
    def safe_logk(flow: torch.Tensor, tin_k: torch.Tensor, upstream: torch.Tensor, downstream: torch.Tensor) -> torch.Tensor:
        # Softplus preserves gradients if an early-epoch prediction temporarily
        # violates pressure ordering.  It changes only the loss calculation.
        disc = upstream.square() - downstream.square()
        safe_disc = nn.functional.softplus(disc / 0.25) * 0.25 + 1e-5
        return torch.log(torch.clamp(flow, min=1.0)) + 0.5 * torch.log(torch.clamp(tin_k, min=300.0)) - 0.5 * torch.log(safe_disc)

    def endpoint_state(self, x24: torch.Tensor, y_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_x = self.raw_x(x24)
        raw_y = self.raw_y(y_norm)
        endpoint = raw_x[:, -1]
        pin = torch.median(endpoint[:, self.p_idx], dim=1).values
        tin_c = torch.median(endpoint[:, self.t_idx], dim=1).values
        tin_k = tin_c + 273.15
        valve_pct = torch.mean(endpoint[:, self.valve_idx], dim=1)
        valve = valve_pct / 100.0
        load = endpoint[:, self.load_idx]
        flow = endpoint[:, self.flow_idx]
        dload = load - raw_x[:, -11, self.load_idx]
        corrected = flow * torch.sqrt(torch.clamp(tin_k, min=300.0)) / torch.clamp(pin, min=1.0)
        return {
            "raw_y": raw_y,
            "pin": pin,
            "tin_c": tin_c,
            "tin_k": tin_k,
            "valve_pct": valve_pct,
            "valve": valve,
            "load": load,
            "flow": flow,
            "dload": dload,
            "corrected": corrected,
            "blade": torch.median(raw_y[:, self.ids["blade"]], dim=1).values,
            "pout": torch.median(raw_y[:, self.ids["pout"]], dim=1).values,
            "pext": raw_y[:, self.ids["ext_p"][0]],
            "pheater": raw_y[:, self.ids["heater_p"][0]],
        }

    def physics_terms(
        self,
        x26: torch.Tensor,
        pred_prev: torch.Tensor,
        pred_curr: torch.Tensor,
        pred_next: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x24 = x26[:, 2:]
        state = self.endpoint_state(x24, pred_next)
        zero = torch.zeros((), dtype=pred_next.dtype, device=pred_next.device)

        blade_f = torch.stack(
            [state["valve"], state["valve"].square(), torch.log(torch.clamp(state["valve"], min=0.01))],
            dim=1,
        )
        logk_blade = self.safe_logk(state["flow"], state["tin_k"], state["pin"], state["blade"])
        z_blade = self.normalise_residual("blade", logk_blade - self.blade_prior(blade_f))
        blade_loss = nn.functional.smooth_l1_loss(z_blade, torch.zeros_like(z_blade), beta=1.0)

        exhaust_f = torch.stack(
            [
                state["corrected"],
                state["valve_pct"],
                state["blade"] / torch.clamp(state["pin"], min=1.0),
                state["load"],
                torch.tanh(state["dload"] / 30.0),
                state["tin_k"],
            ],
            dim=1,
        )
        logk_exhaust = self.safe_logk(state["flow"], state["tin_k"], state["blade"], state["pout"])
        z_exhaust = self.normalise_residual("exhaust", logk_exhaust - self.exhaust_prior(exhaust_f))
        ext_f = torch.stack(
            [
                state["valve"],
                state["valve"].square(),
                torch.log(torch.clamp(state["valve"], min=0.01)),
                state["corrected"],
                state["load"],
                state["pin"],
                state["tin_c"],
            ],
            dim=1,
        )
        logk_ext = self.safe_logk(state["flow"], state["tin_k"], state["pin"], state["pext"])
        z_ext = self.normalise_residual("extraction", logk_ext - self.extraction_prior(ext_f))
        stodola_loss = 0.5 * (
            nn.functional.smooth_l1_loss(z_exhaust, torch.zeros_like(z_exhaust), beta=1.0)
            + nn.functional.smooth_l1_loss(z_ext, torch.zeros_like(z_ext), beta=1.0)
        )

        eta_losses = []
        eta_z: dict[str, torch.Tensor] = {}
        for name, p_key, t_key in (
            ("vhp", "pout", "vhp_t"),
            ("vertical", "pout", "vertical_t"),
            ("extraction", "pext", "ext_t"),
            ("heater", "pheater", "heater_t"),
        ):
            pressure = state[p_key]
            temp_c = torch.median(state["raw_y"][:, self.ids[t_key]], dim=1).values
            # Clamps are numerical guards inside Loss, not output overrides.
            p_safe = torch.clamp(pressure, min=0.2, max=30.0)
            t_safe = torch.clamp(temp_c + 273.15, min=450.0, max=900.0)
            eta = self.if97.efficiency(state["pin"], state["tin_k"], p_safe, t_safe)
            prior_f = torch.stack([pressure / torch.clamp(state["pin"], min=1.0), state["corrected"]], dim=1)
            z = self.normalise_residual(f"eta_{name}", eta - self.eta_priors[name](prior_f))
            eta_z[name] = z
            eta_losses.append(nn.functional.smooth_l1_loss(z, torch.zeros_like(z), beta=1.0))
        if97_loss = torch.stack(eta_losses).mean() if eta_losses else zero

        raw_prev, raw_curr, raw_next = self.raw_y(pred_prev), self.raw_y(pred_curr), self.raw_y(pred_next)
        pipe_losses, pipe_z = [], []
        for output_idx, names in self.pipe_specs.items():
            a, bias, center, scale = (getattr(self, n) for n in names)
            steam_ids = self.ids["ext_t"] if output_idx in self.ids["pipe_ext"] else self.ids["vertical_t"]
            steam_curr = torch.median(raw_curr[:, steam_ids], dim=1).values
            derivative = (raw_next[:, output_idx] - raw_prev[:, output_idx]) / 2.0
            driving = steam_curr - raw_curr[:, output_idx]
            z = (derivative - a * driving - bias - center) / scale
            pipe_z.append(z)
            pipe_losses.append(nn.functional.smooth_l1_loss(z, torch.zeros_like(z), beta=1.0))
        pipe_loss = torch.stack(pipe_losses).mean() if pipe_losses else zero
        return {
            "stodola": stodola_loss,
            "blade": blade_loss,
            "if97": if97_loss,
            "pipe": pipe_loss,
            "z_exhaust": z_exhaust,
            "z_extraction": z_ext,
            "z_blade": z_blade,
            "z_pipe_mean": torch.stack(pipe_z).mean(0),
            **{f"z_eta_{name}": value for name, value in eta_z.items()},
        }


def forward_triplet(model: V21SoftPIGRU, x26: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Each invocation is still the exact same 24-minute predictor used at inference.
    prev = model(x26[:, 0:24])
    curr = model(x26[:, 1:25])
    nxt = model(x26[:, 2:26])
    return prev, curr, nxt


def synthetic_collocation(x26: torch.Tensor) -> torch.Tensor:
    """Create unlabeled, physically coherent interpolation samples.

    Convex interpolation between two real 26-minute operating trajectories
    stays near the observed manifold.  Small common-mode boundary shifts widen
    pressure, temperature, valve/load, and flow coverage without inventing Y.
    """

    perm = torch.randperm(len(x26), device=x26.device)
    alpha = torch.empty((len(x26), 1, 1), device=x26.device).uniform_(0.15, 0.85)
    synth = alpha * x26 + (1.0 - alpha) * x26[perm]
    # Values are standardized: common shifts preserve redundant-sensor consensus.
    pressure_shift = torch.randn((len(x26), 1, 1), device=x26.device) * 0.12
    temperature_shift = torch.randn((len(x26), 1, 1), device=x26.device) * 0.12
    synth[:, :, :6] += pressure_shift
    synth[:, :, 6:12] += temperature_shift
    common_boundary = torch.randn((len(x26), 1, 1), device=x26.device) * 0.08
    synth[:, :, 12:16] += common_boundary
    return synth


def gradient_diagnostics(
    data_loss: torch.Tensor,
    physics: dict[str, torch.Tensor],
    families: tuple[str, ...],
    model: V21SoftPIGRU,
) -> tuple[dict[str, dict[str, float]], float]:
    params = [p for p in model.predictor.parameters() if p.requires_grad]
    data_grads = torch.autograd.grad(data_loss, params, retain_graph=True, allow_unused=True)
    data_norm_sq = sum(torch.sum(g.float().square()) for g in data_grads if g is not None)
    data_norm = torch.sqrt(data_norm_sq + 1e-18)
    stats: dict[str, dict[str, float]] = {}
    for family in families:
        p_grads = torch.autograd.grad(physics[family], params, retain_graph=True, allow_unused=True)
        p_norm_sq = sum(torch.sum(g.float().square()) for g in p_grads if g is not None)
        p_norm = torch.sqrt(p_norm_sq + 1e-18)
        dot = sum(
            torch.sum(gd.float() * gp.float())
            for gd, gp in zip(data_grads, p_grads)
            if gd is not None and gp is not None
        )
        cosine = dot / (data_norm * p_norm + 1e-18)
        stats[family] = {"grad_norm": float(p_norm.detach()), "cosine": float(cosine.detach())}
    return stats, float(data_norm.detach())


def loss_parts(
    model: V21SoftPIGRU,
    x26: torch.Tensor,
    yb: torch.Tensor,
    profile: Profile,
    weights: dict[str, float],
    ramp: float,
    need_gradient_stats: bool = False,
    apply_synthetic: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, float]], float]:
    if profile.families:
        prev, curr, pred = forward_triplet(model, x26)
        physics = model.physics_terms(x26, prev, curr, pred)
    else:
        pred = model(x26[:, 2:])
        physics = {}
    data = torch.mean((pred - yb).square())
    total = data
    for family in profile.families:
        total = total + ramp * weights[family] * physics[family]
    synthetic_loss = torch.zeros((), device=x26.device)
    if profile.synthetic and ramp > 0 and apply_synthetic:
        # Physics collocation does not need to duplicate every supervised row.
        # A deterministic-size 25% subset keeps GPU use proportional while the
        # random subset changes every batch/epoch and covers the full manifold.
        n_collocation = max(64, len(x26) // 4)
        source_ids = torch.randperm(len(x26), device=x26.device)[:n_collocation]
        sx = synthetic_collocation(x26[source_ids])
        sp, sc, sn = forward_triplet(model, sx)
        sphysics = model.physics_terms(sx, sp, sc, sn)
        synthetic_loss = torch.stack([sphysics[f] for f in profile.families]).mean()
        total = total + ramp * 0.15 * sum(weights[f] for f in profile.families) * synthetic_loss
    stats: dict[str, dict[str, float]] = {}
    data_grad_norm = np.nan
    if need_gradient_stats and profile.families:
        stats, data_grad_norm = gradient_diagnostics(data, physics, profile.families, model)
    parts = {"total": total, "data": data, "synthetic": synthetic_loss, **{f: physics[f] for f in profile.families}}
    return total, parts, stats, data_grad_norm


@torch.inference_mode()
def validation_loss(
    model: V21SoftPIGRU,
    data: DataLoader,
    profile: Profile,
    weights: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for xb, yb in data:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        _, parts, _, _ = loss_parts(model, xb, yb, profile, weights, 1.0)
        for name, value in parts.items():
            sums[name] = sums.get(name, 0.0) + float(value) * len(xb)
        count += len(xb)
    return {name: value / max(count, 1) for name, value in sums.items()}


def adapt_weights(
    weights: dict[str, float],
    stats: dict[str, dict[str, float]],
    data_norm: float,
) -> dict[str, float]:
    updated = dict(weights)
    for family, item in stats.items():
        target_ratio = 0.15
        # A negative cosine means the equation is currently asking the shared
        # GRU parameters to move against the data objective.  Reduce, rather
        # than completely remove, that family until the conflict disappears.
        if item["cosine"] < 0.0:
            target_ratio *= 0.25
        target = target_ratio * data_norm / max(item["grad_norm"], 1e-12)
        target = float(np.clip(target, 1e-5, 0.03))
        updated[family] = 0.8 * weights[family] + 0.2 * target
    return updated


def train_selection(
    profile: Profile,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    model_args: tuple[Any, ...],
    device: torch.device,
    seed: int,
    max_epochs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, float], float]:
    seed_everything(seed)
    model = V21SoftPIGRU(*model_args).to(device)
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=2e-6)
    train_data = loader(fit[0], fit[1], 2048, True)
    val_data = loader(val[0], val[1], 4096, False)
    weights = {family: BASE_WEIGHTS[family] for family in profile.families}
    history, grad_rows = [], []
    best, best_epoch, best_state, stale = math.inf, 0, None, 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        count = 0
        # Twenty data-only warmup epochs, then a thirty-epoch curriculum.
        ramp = 0.0 if epoch <= 20 else min(1.0, (epoch - 20) / 30.0)
        synthetic_batches = 0
        for batch_idx, (xb, yb) in enumerate(train_data):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            diagnose = bool(profile.families) and batch_idx == 0 and (epoch == 1 or epoch % 10 == 0)
            use_synthetic = profile.synthetic and batch_idx % 2 == 0
            total, parts, stats, data_norm = loss_parts(
                model, xb, yb, profile, weights, ramp, diagnose, apply_synthetic=use_synthetic
            )
            synthetic_batches += int(use_synthetic and ramp > 0)
            if diagnose:
                for family, item in stats.items():
                    grad_rows.append(
                        {
                            "phase": "selection",
                            "epoch": epoch,
                            "family": family,
                            "data_grad_norm": data_norm,
                            "physics_grad_norm": item["grad_norm"],
                            "cosine": item["cosine"],
                            "weight_before": weights[family],
                            "weighted_physics_grad_ratio": weights[family] * item["grad_norm"] / max(data_norm, 1e-12),
                        }
                    )
                if profile.adaptive and epoch > 20:
                    weights = adapt_weights(weights, stats, data_norm)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 5.0)
            optimizer.step()
            for name, value in parts.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(xb)
            count += len(xb)
        val_loss = validation_loss(model, val_data, profile, weights, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_total": sums["total"] / count,
            "train_data": sums["data"] / count,
            "train_synthetic": sums.get("synthetic", 0.0) / count,
            "val_total": val_loss["total"],
            "val_data": val_loss["data"],
            "lr": optimizer.param_groups[0]["lr"],
            "physics_ramp": ramp,
            **{f"weight_{family}": weights[family] for family in profile.families},
            "synthetic_batch_fraction": synthetic_batches / max(len(train_data), 1),
        }
        for family in profile.families:
            row[f"train_{family}"] = sums[family] / count
            row[f"val_{family}"] = val_loss[family]
        history.append(row)
        scheduler.step(val_loss["data"])
        if val_loss["data"] < best * (1.0 - 1e-4):
            best, best_epoch, best_state, stale = float(val_loss["data"]), epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch >= 70 and stale >= 35:
            break
    del best_state  # selection chooses the epoch; a clean full-window retrain follows.
    return history, grad_rows, best_epoch, weights, time.perf_counter() - started


def train_full(
    profile: Profile,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    model_args: tuple[Any, ...],
    selection_history: list[dict[str, Any]],
    best_epoch: int,
    final_weights: dict[str, float],
    device: torch.device,
    seed: int,
) -> tuple[V21SoftPIGRU, list[dict[str, Any]], float]:
    seed_everything(seed)
    model = V21SoftPIGRU(*model_args).to(device)
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    data = loader(train[0], train[1], 2048, True)
    rows = []
    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        lr = selection_history[min(epoch - 1, len(selection_history) - 1)]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        ramp = 0.0 if epoch <= 20 else min(1.0, (epoch - 20) / 30.0)
        # Replay the selected weight curriculum; adaptive profiles use the
        # leakage-safe selection trajectory rather than recalibrating on full data.
        weights = {
            family: float(selection_history[min(epoch - 1, len(selection_history) - 1)].get(f"weight_{family}", final_weights[family]))
            for family in profile.families
        }
        model.train()
        sums: dict[str, float] = {}
        count = 0
        synthetic_batches = 0
        for batch_idx, (xb, yb) in enumerate(data):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_synthetic = profile.synthetic and batch_idx % 2 == 0
            total, parts, _, _ = loss_parts(
                model, xb, yb, profile, weights, ramp, apply_synthetic=use_synthetic
            )
            synthetic_batches += int(use_synthetic and ramp > 0)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 5.0)
            optimizer.step()
            for name, value in parts.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(xb)
            count += len(xb)
        rows.append(
            {
                "epoch": epoch,
                **{name: value / count for name, value in sums.items()},
                "lr": lr,
                "physics_ramp": ramp,
                **{f"weight_{family}": weights[family] for family in profile.families},
                "synthetic_batch_fraction": synthetic_batches / max(len(data), 1),
            }
        )
    return model, rows, time.perf_counter() - started


@torch.inference_mode()
def predict(model: V21SoftPIGRU, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    for start in range(0, len(x), 8192):
        xb = torch.from_numpy(np.ascontiguousarray(x[start : start + 8192])).to(device)
        rows.append(model(xb[:, -WINDOW:]).cpu().numpy())
    return np.concatenate(rows)


def plot_losses(protocol: str, profile: str, seed: int, selection: list[dict[str, Any]], full: list[dict[str, Any]]) -> None:
    configure_font()
    s, f = pd.DataFrame(selection), pd.DataFrame(full)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(s.epoch, s.train_data, label="训练数据 Loss")
    axes[0].plot(s.epoch, s.val_data, label="验证数据 Loss")
    for family in PROFILES[profile].families:
        axes[0].plot(s.epoch, s[f"train_{family}"], alpha=0.7, label=f"物理残差 {family}")
    axes[1].plot(f.epoch, f.data, label="完整训练数据 Loss")
    for family in PROFILES[profile].families:
        axes[1].plot(f.epoch, f[family], alpha=0.7, label=f"物理残差 {family}")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_title("前80%拟合 / 后20%验证")
    axes[1].set_title("完整10天重训练")
    fig.suptitle(f"{protocol} | {profile} | seed={seed}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"loss_{protocol}_{profile}_seed{seed}.png", dpi=180)
    plt.close(fig)


def run_one(
    protocol: str,
    profile_name: str,
    seed: int,
    metadata: dict[str, Any],
    calibration: dict[str, Any],
    device: torch.device,
    max_epochs: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    profile = PROFILES[profile_name]
    d = np.load(CACHE / f"{protocol}.npz")
    train_x, train_y, train_time = d["train_x"].astype(float), d["train_y"].astype(float), d["train_time"]
    test_x, test_y, test_time = d["test_x"].astype(float), d["test_y"].astype(float), d["test_time"]
    inputs, targets = list(metadata["baseline_inputs"]), list(metadata["targets"])
    split = split_index(train_time)

    xs_sel, ys_sel = StandardScaler().fit(train_x[:split]), StandardScaler().fit(train_y[:split])
    fit = make_samples(train_x[:split], train_y[:split], train_time[:split], xs_sel, ys_sel)
    val = make_samples(train_x[split:], train_y[split:], train_time[split:], xs_sel, ys_sel)
    selection, grad_rows, best_epoch, final_weights, selection_seconds = train_selection(
        profile,
        fit,
        val,
        (xs_sel, ys_sel, inputs, targets, calibration),
        device,
        seed,
        max_epochs,
    )

    xs, ys = StandardScaler().fit(train_x), StandardScaler().fit(train_y)
    train = make_samples(train_x, train_y, train_time, xs, ys)
    test_eval = make_samples(test_x, test_y, test_time, xs, ys, window=WINDOW)
    train_eval = make_samples(train_x, train_y, train_time, xs, ys, window=WINDOW)
    model, full_history, full_seconds = train_full(
        profile,
        train,
        (xs, ys, inputs, targets, calibration),
        selection,
        best_epoch,
        final_weights,
        device,
        seed,
    )
    train_pred = ys.inverse_transform(predict(model, train_eval[0], device))
    test_pred = ys.inverse_transform(predict(model, test_eval[0], device))
    train_actual = ys.inverse_transform(train_eval[1])
    test_actual = ys.inverse_transform(test_eval[1])
    points = pd.concat(
        [
            core.point_metrics(protocol, profile_name, "train", train_actual, train_pred, targets, metadata["target_metadata"], train_actual),
            core.point_metrics(protocol, profile_name, "test", test_actual, test_pred, targets, metadata["target_metadata"], train_actual),
        ],
        ignore_index=True,
    )
    points.insert(2, "seed", seed)
    test_points, train_points = points[points.split == "test"], points[points.split == "train"]
    pressure_tags = BLADE + POUT + EXT_P + HEATER_P
    steam_temp_tags = VHP_T + VERTICAL_T + EXT_T + HEATER_T
    pipe_tags = PIPE_EXT + PIPE_RH1
    summary = {
        "protocol": protocol,
        "profile": profile_name,
        "seed": seed,
        "architecture": (
            f"24x{len(inputs)} -> GRU(32,1 layer) -> LayerNorm -> "
            f"Dropout(0.05) -> Linear({len(targets)})"
        ),
        "trainable_parameters": sum(p.numel() for p in model.predictor.parameters() if p.requires_grad),
        "physics_trainable_parameters": 0,
        "all_21_direct_outputs": len(targets) == 21,
        "all_outputs_direct_predictions": True,
        "output_count": len(targets),
        "physics_only_in_loss": bool(profile.families),
        "physical_output_override": False,
        "adaptive_weights": profile.adaptive,
        "unlabeled_synthetic_collocation": profile.synthetic,
        "physics_families": "+".join(profile.families) if profile.families else "none",
        "history_prior": calibration["history_label"],
        "best_epoch": best_epoch,
        "best_val_data_loss": min(row["val_data"] for row in selection),
        "train_samples": len(train[0]),
        "test_samples": len(test_eval[0]),
        "macro_train_r2": float(train_points.r2.mean()),
        "macro_test_r2": float(test_points.r2.mean()),
        "macro_r2_gap": float(train_points.r2.mean() - test_points.r2.mean()),
        "pressure7_test_r2": float(test_points[test_points.target_tag.isin(pressure_tags)].r2.mean()),
        "steam_temperature7_test_r2": float(test_points[test_points.target_tag.isin(steam_temp_tags)].r2.mean()),
        "pipe_temperature7_test_r2": float(test_points[test_points.target_tag.isin(pipe_tags)].r2.mean()),
        "macro_test_rmse": float(test_points.rmse.mean()),
        "macro_test_mae": float(test_points.mae.mean()),
        "selection_seconds": selection_seconds,
        "full_seconds": full_seconds,
        "device": str(device),
        "uses_test_truth_for_training_or_selection": False,
        "uses_target_history_as_input": False,
    }

    run_dir = MODELS / protocol / profile_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "profile": profile_name,
            "inputs": inputs,
            "targets": targets,
            "x_mean": xs.mean_,
            "x_scale": xs.scale_,
            "y_mean": ys.mean_,
            "y_scale": ys.scale_,
            "calibration": calibration,
            "best_epoch": best_epoch,
            "architecture": summary["architecture"],
        },
        run_dir / "model.pt",
    )
    pd.DataFrame(selection).to_csv(run_dir / "selection_loss.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(full_history).to_csv(run_dir / "full_train_loss.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(grad_rows).assign(protocol=protocol, profile=profile_name, seed=seed).to_csv(
        run_dir / "gradient_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        PREDICTIONS / f"{protocol}_{profile_name}_seed{seed}.npz",
        train_time=train_eval[2],
        train_actual=train_actual,
        train_pred=train_pred,
        test_time=test_eval[2],
        test_actual=test_actual,
        test_pred=test_pred,
        target_tags=np.asarray(targets),
    )
    plot_losses(protocol, profile_name, seed, selection, full_history)
    return points, summary, pd.DataFrame(grad_rows).assign(protocol=protocol, profile=profile_name, seed=seed)


def aggregate() -> None:
    if not (OUT / "model_summary.csv").exists():
        return
    summary = pd.read_csv(OUT / "model_summary.csv")
    points = pd.read_csv(OUT / "metrics_by_point.csv")
    comp = summary.groupby(["protocol", "profile"], as_index=False).agg(
        seeds=("seed", "nunique"),
        macro_test_r2_mean=("macro_test_r2", "mean"),
        macro_test_r2_std=("macro_test_r2", "std"),
        pressure7_r2=("pressure7_test_r2", "mean"),
        steam_temperature7_r2=("steam_temperature7_test_r2", "mean"),
        pipe_temperature7_r2=("pipe_temperature7_test_r2", "mean"),
        train_test_gap=("macro_r2_gap", "mean"),
        best_epoch=("best_epoch", "mean"),
    )
    comp.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    point_test = points[points.split == "test"]
    point_agg = point_test.groupby(["protocol", "algorithm", "target_tag", "name_cn", "unit"], as_index=False).agg(
        seeds=("seed", "nunique"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        train_min=("train_actual_min", "first"),
        train_max=("train_actual_max", "first"),
    )
    point_agg.to_csv(OUT / "metrics_by_point_aggregate.csv", index=False, encoding="utf-8-sig")
    print(comp.to_string(index=False), flush=True)


def append_csv(path: Path, frame: pd.DataFrame, keys: list[str], overwrite_keys: set[tuple[Any, ...]]) -> None:
    if path.exists():
        old = pd.read_csv(path)
        if len(old):
            mask = old.apply(lambda row: tuple(row[k] for k in keys) in overwrite_keys, axis=1)
            old = old[~mask]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_architecture(metadata: dict[str, Any]) -> None:
    text = f"""# V21 同架构 GRU / PINN-GRU 说明

## 唯一预测结构

`24×{len(metadata['baseline_inputs'])} → 单层 GRU(hidden=32) → LayerNorm(32) → Dropout(0.05) → Linear(32, {len(metadata['targets'])})`

- GRU 与所有 PINN 配置的可训练结构、参数量、输入窗口和 21 个输出完全相同。
- 21 个输出全部由神经网络直接给出；推理时只执行 `X → GRU → Y_hat`。
- 不使用真实历史输出作为输入，不串联下游真实值，不让物理解覆盖、融合或门控预测值。
- PINN 的区别仅为训练 Loss 中增加通过真实 DCS 残差审计的无量纲方程残差。

## Physics Loss 分级

1. `PINN_STODOLA_FIXED`：排汽段与一级抽汽段 Stodola/Flügel（两个年份均通过）。
2. `PINN_STODOLA_IF97_FIXED`：再加入 IF97 等熵效率一致性（2025 为 conditional）。
3. `PINN_FULL_FIXED`：再加入叶片级 Stodola 与 7 点管壁一阶热惯性，使 21 点全部受物理关系约束。
4. `PINN_FULL_ADAPTIVE`：结构不变，依据数据/物理梯度范数及余弦冲突调整权重。
5. `PINN_FULL_ADAPTIVE_SYNTH`：再增加无标签、实测轨迹凸插值的宽边界物理配点；合成样本不进入数据 Loss。

20 epoch 只学习数据关系，随后 30 epoch 逐步引入物理项。方程是否进入 Loss 以 `42_v21_measured_physics_audit` 为依据；被拒绝的全链压力有序约束和 2024 抽汽压降不进入主模型。
"""
    (OUT / "V21_网络结构与PhysicsLoss.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", nargs="+", default=["10d100d_2024", "10d100d_2025"])
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-epochs", type=int, default=260)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-physics-calibration", action="store_true")
    args = parser.parse_args()
    configure_compute(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8-sig"))
    inputs, targets = list(metadata["baseline_inputs"]), list(metadata["targets"])
    write_architecture(metadata)
    calibrations = {}
    for year in sorted({int(p.rsplit("_", 1)[1]) for p in args.protocols}):
        calibrations[year] = load_or_build_calibration(
            year, inputs, targets, rebuild=args.rebuild_physics_calibration
        )

    existing = pd.read_csv(OUT / "model_summary.csv") if (OUT / "model_summary.csv").exists() else pd.DataFrame()
    summaries, points, grads = [], [], []
    for protocol in args.protocols:
        year = int(protocol.rsplit("_", 1)[1])
        for profile in args.profiles:
            for seed in args.seeds:
                done = (
                    not existing.empty
                    and ((existing.protocol == protocol) & (existing.profile == profile) & (existing.seed == seed)).any()
                )
                if done and not args.overwrite:
                    print(f"[skip] {protocol} {profile} seed={seed}", flush=True)
                    continue
                print(f"[run] {protocol} {profile} seed={seed}", flush=True)
                point_frame, summary, grad_frame = run_one(
                    protocol, profile, seed, metadata, calibrations[year], device, args.max_epochs
                )
                summaries.append(summary)
                points.append(point_frame)
                if len(grad_frame):
                    grads.append(grad_frame)
                key = {(protocol, profile, seed)}
                append_csv(OUT / "model_summary.csv", pd.DataFrame([summary]), ["protocol", "profile", "seed"], key)
                append_csv(OUT / "metrics_by_point.csv", point_frame, ["protocol", "algorithm", "seed"], key)
                if len(grad_frame):
                    append_csv(OUT / "gradient_diagnostics.csv", grad_frame, ["protocol", "profile", "seed"], key)
                aggregate()
    aggregate()


if __name__ == "__main__":
    main()
