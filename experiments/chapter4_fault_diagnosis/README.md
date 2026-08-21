# Chapter 4 — KG-GCN fault diagnosis

The chapter follows a single reproducible chain:

`healthy DCS state → health-model residual → train-only residual standardization → 60 min node window → semi-physical fault injection → equipment KG → masked-pretraining KG-GCN → fault classification`.

## Paper-facing cases

- VHP main case: Macro-F1 98.46% ± 0.11%.
- H1 high-pressure heater: 95.49% ± 0.30%.
- L6 low-pressure heater: 98.09% ± 0.17%.
- BFPT feed-pump turbine: 96.00% ± 0.47%.
- CND1 condenser: 97.40% ± 0.76%.
- VHP–RH1 system extension: 94.79% ± 0.50%.

H2/H3/H4 remain same-type repeatability checks and are not counted as different equipment types in the paper.

## Integrity boundary

Faults are mechanism-informed semi-physical injections on real healthy residual backgrounds, not labeled historical plant faults. Train/validation/test dates are mutually exclusive, duplicate windows are audited, and fault-signature generation does not read the KG adjacency matrix.
