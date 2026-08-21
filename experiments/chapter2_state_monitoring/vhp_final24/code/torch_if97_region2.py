from __future__ import annotations

"""Differentiable IF97 Region-2 water/steam properties for PyTorch.

The coefficients are the official IAPWS-IF97 Region-2 coefficients exposed by
the installed ``iapws`` package.  All equations below follow IF97 Eq. 15-17.
The VHP inlet and exhaust states used by this project are superheated steam and
fall inside Region 2.  The module returns h, s and cp and provides a fixed-step
Newton solve for the isentropic outlet temperature; gradients propagate through
both the property equations and the unrolled solve.
"""

import numpy as np
import torch
from iapws import _iapws97Constants as Const
from torch import nn


R_KJ_KG_K = 0.461526


class IF97Region2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # iapws<=1.5.4 exposed the Region-2 residual coefficients as
        # Region2_Ir/Jr/nr.  iapws 1.5.5 renamed the same official arrays to
        # Region2_Li/Lj/n.  Support both public layouts so a clean V21/V22
        # environment does not fail before training.
        ir = Const.Region2_Ir if hasattr(Const, "Region2_Ir") else Const.Region2_Li
        jr = Const.Region2_Jr if hasattr(Const, "Region2_Jr") else Const.Region2_Lj
        nr = Const.Region2_nr if hasattr(Const, "Region2_nr") else Const.Region2_n
        arrays = {
            "ir": ir,
            "jr": jr,
            "nr": nr,
            "j0": Const.Region2_cp0_Jo,
            "n0": Const.Region2_cp0_no,
        }
        for name, value in arrays.items():
            self.register_buffer(name, torch.tensor(np.asarray(value), dtype=torch.float32), persistent=True)

    def properties(self, pressure_mpa: torch.Tensor, temperature_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = torch.clamp(pressure_mpa.float(), min=1e-4, max=100.0)
        t = torch.clamp(temperature_k.float(), min=273.16, max=1073.15)
        tau = 540.0 / t
        pi = p

        tau_e = tau.unsqueeze(-1)
        pi_e = pi.unsqueeze(-1)
        delta = tau_e - 0.5

        go = torch.log(pi) + torch.sum(self.n0 * torch.pow(tau_e, self.j0), dim=-1)
        got = torch.sum(self.n0 * self.j0 * torch.pow(tau_e, self.j0 - 1.0), dim=-1)
        gott = torch.sum(self.n0 * self.j0 * (self.j0 - 1.0) * torch.pow(tau_e, self.j0 - 2.0), dim=-1)

        base = self.nr * torch.pow(pi_e, self.ir) * torch.pow(delta, self.jr)
        gr = torch.sum(base, dim=-1)
        grt = torch.sum(self.nr * self.jr * torch.pow(pi_e, self.ir) * torch.pow(delta, self.jr - 1.0), dim=-1)
        grtt = torch.sum(
            self.nr * self.jr * (self.jr - 1.0) * torch.pow(pi_e, self.ir) * torch.pow(delta, self.jr - 2.0),
            dim=-1,
        )

        h = tau * (got + grt) * R_KJ_KG_K * t
        s = R_KJ_KG_K * (tau * (got + grt) - (go + gr))
        cp = -R_KJ_KG_K * tau.square() * (gott + grtt)
        return h, s, cp

    def isentropic_outlet(
        self,
        inlet_pressure_mpa: torch.Tensor,
        inlet_temperature_k: torch.Tensor,
        outlet_pressure_mpa: torch.Tensor,
        iterations: int = 7,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_in, s_in, _ = self.properties(inlet_pressure_mpa, inlet_temperature_k)
        ratio = torch.clamp(outlet_pressure_mpa / torch.clamp(inlet_pressure_mpa, min=1e-4), 0.02, 0.999)
        # Ideal-gas-like initialization is only a numerical starting point; the
        # converged state is determined by the exact IF97 entropy equation.
        t_is = torch.clamp(inlet_temperature_k * torch.pow(ratio, 0.22), min=400.0, max=1000.0)
        for _ in range(iterations):
            h_guess, s_guess, cp_guess = self.properties(outlet_pressure_mpa, t_is)
            ds_dt = torch.clamp(cp_guess / t_is, min=1e-5)
            step = torch.clamp((s_guess - s_in) / ds_dt, min=-80.0, max=80.0)
            t_is = torch.maximum(t_is - step, torch.full_like(t_is, 350.0))
            t_is = torch.minimum(t_is, inlet_temperature_k)
        h_is, _, _ = self.properties(outlet_pressure_mpa, t_is)
        return h_in, h_is, t_is

    def efficiency(
        self,
        inlet_pressure_mpa: torch.Tensor,
        inlet_temperature_k: torch.Tensor,
        outlet_pressure_mpa: torch.Tensor,
        outlet_temperature_k: torch.Tensor,
    ) -> torch.Tensor:
        h_in, h_is, _ = self.isentropic_outlet(inlet_pressure_mpa, inlet_temperature_k, outlet_pressure_mpa)
        h_out, _, _ = self.properties(outlet_pressure_mpa, outlet_temperature_k)
        denominator = torch.clamp(h_in - h_is, min=1.0)
        return (h_in - h_out) / denominator

    def temperature_from_enthalpy(
        self,
        pressure_mpa: torch.Tensor,
        target_enthalpy_kjkg: torch.Tensor,
        initial_temperature_k: torch.Tensor,
        lower_temperature_k: torch.Tensor | float = 350.0,
        upper_temperature_k: torch.Tensor | float = 1073.15,
        iterations: int = 8,
    ) -> torch.Tensor:
        """Invert Region-2 h(p,T) with an unrolled Newton solve.

        At fixed pressure, ``dh/dT = cp``.  Clamped Newton steps keep the
        VHP effective expansion solve stable while preserving gradients.
        """

        t = torch.clamp(initial_temperature_k.float(), min=350.0, max=1073.15)
        lower = torch.as_tensor(lower_temperature_k, dtype=t.dtype, device=t.device)
        upper = torch.as_tensor(upper_temperature_k, dtype=t.dtype, device=t.device)
        for _ in range(iterations):
            h, _, cp = self.properties(pressure_mpa, t)
            step = torch.clamp((h - target_enthalpy_kjkg) / torch.clamp(cp, min=0.2), min=-80.0, max=80.0)
            t = torch.minimum(torch.maximum(t - step, lower), upper)
        return t

    def outlet_temperature_from_efficiency(
        self,
        inlet_pressure_mpa: torch.Tensor,
        inlet_temperature_k: torch.Tensor,
        outlet_pressure_mpa: torch.Tensor,
        efficiency: torch.Tensor,
    ) -> torch.Tensor:
        """Effective-control-volume outlet temperature from isentropic efficiency."""

        h_in, h_is, t_is = self.isentropic_outlet(inlet_pressure_mpa, inlet_temperature_k, outlet_pressure_mpa)
        eta = torch.clamp(efficiency, min=0.01, max=0.999)
        h_out = h_in - eta * (h_in - h_is)
        initial = t_is + (1.0 - eta) * (inlet_temperature_k - t_is)
        return self.temperature_from_enthalpy(
            outlet_pressure_mpa,
            h_out,
            initial,
            lower_temperature_k=torch.clamp(t_is - 20.0, min=350.0),
            upper_temperature_k=inlet_temperature_k,
        )
