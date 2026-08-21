# Experiment index

This repository separates the paper experiments by manuscript role rather than by development version.

## Chapter 2 — state monitoring

- `chapter2_state_monitoring/vhp_final24/`: final VHP 24-state reporting package for the 2024-05→06 protocol, including four-model prediction arrays, pointwise metrics, loss/config files and reproducible scripts.
- `chapter2_state_monitoring/non_vhp_representatives_20260822/`: H1, deaerator and condenser representative state-estimation audits used to support cross-device residual construction.

## Chapter 4 — fault diagnosis

- `chapter4_fault_diagnosis/device_transfer_20260822/`: device-level KG-GCN transfer cases for H1, L6, BFPT and CND1.
- `chapter4_fault_diagnosis/vhp_rh1_system_extension/`: 36-node VHP–RH1 system extension.
- `chapter4_fault_diagnosis/methodology_audits/gap_readout/`: Global Average Pooling versus node-identity-preserving readout audit.

## Shared protocol

- `shared_protocols/`: common residual-window, semi-physical fault injection, split and integrity rules.

## Data boundary

The 7 GB plant-wide DCS export is intentionally excluded. Curated NPZ artifacts required to reproduce figures and tables are stored with Git LFS. Generated caches, duplicate figures, temporary workbooks and local absolute paths are excluded.
