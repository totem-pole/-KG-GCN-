# Level-C v08: Raw DCS State vs GRU Healthy Residual

## Purpose
This experiment isolates the contribution of Chapter 2 healthy-state modeling to Chapter 5 KG-GCN diagnosis.

The train/validation/test dates, 24 VHP DCS nodes, Level-C fault signatures, temporal profiles, class balance, and KG-GCN architecture are unchanged from v07. The only changed factor is the diagnostic input baseline:

1. **Raw-DCS-state**: healthy DCS state window + exactly the same physical fault deviation;
2. **GRU-Residual**: measured-minus-healthy-GRU residual window + fault deviation.

Both inputs use training-only per-node standardization.

## Three-seed result

| Input | Accuracy mean ± std | Macro-F1 mean ± std |
|---|---:|---:|
| Raw DCS state | 0.85952 ± 0.02398 | 0.86030 ± 0.02414 |
| **GRU healthy residual** | **0.98452 ± 0.00103** | **0.98456 ± 0.00105** |

Residual input improves Macro-F1 by **0.12425 (12.43 percentage points)** and substantially reduces random-seed variation.

Per seed, Raw-DCS-state KG-GCN Macro-F1:
- seed 123: 0.87019
- seed 101: 0.83279
- seed 202: 0.87793

This result supports the paper-level chain:

`healthy-state GRU -> residual -> KG parameter graph -> masked GCN -> fault classification`

## Interpretation
Raw DCS states retain normal load/boundary-driven variation. The GRU residual removes a large fraction of this predictable healthy variation before fault diagnosis, making the remaining abnormal deviation more concentrated and easier to associate with KG parameter paths.

This is still a Level-C semi-physical validation and must not be described as field-fault accuracy.
