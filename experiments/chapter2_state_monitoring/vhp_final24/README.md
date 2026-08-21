# VHP final 24-state package

## Paper protocol

- Train: 2024-05; independent test: 2024-06.
- Inputs: 16 causal boundary variables.
- Models: ANN, TCN, iTransformer-small and GRU-RNN.
- Final reported states: 24.
- Macro R²: ANN 0.908731, TCN 0.939679, iTransformer-small 0.935611, GRU-RNN 0.951118.

## Version fact that must not be lost

The existing V24 networks were trained with 27 outputs. V25 removed CT132A, CT151A and CT152A without retraining, producing the final 24-state evaluation tables and plots. Therefore the original prediction NPZ files still contain 27 columns. Apply `final_24_outputs.json` before paper plotting. Do not describe the archived weights as a native 24-output head.

## Files

- `predictions/`: original-scale 2024 monthly predictions for all four models.
- `metrics/`: final 24-state pointwise metrics and frozen reporting rules.
- `models/`: configurations, weights and selection/full-training losses.
- `cache/`: 2024 monthly engineering-scale inputs/targets and protocol metadata.
- `code/`: training and analysis code recovered from V24.

Large binary artifacts are managed through Git LFS.
