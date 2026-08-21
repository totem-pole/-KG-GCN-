# Shared experimental protocol

## Residual construction

For equipment state `i`, the health residual is

`e_i(t) = y_i(t) - yhat_i(t)`.

It is standardized using healthy training data only:

`r_i(t) = (e_i(t) - mean_i_train) / std_i_train`.

The graph input is a fixed-order node matrix containing a 60 min residual window.

## Fault generation

- Normal samples preserve the real healthy residual background.
- Fault samples add a frozen mechanism-informed spatial signature and a temporal profile.
- Main visible-fault severity is 2–3 healthy-residual standard deviations.
- Train, validation and test use independent random seeds.
- Weak-severity and unseen-step profiles are separate stress tests and never mixed with the main IID score.
- Fault signatures and KG adjacency are defined independently.

## Selection and testing

- Candidate equipment is selected on validation Macro-F1 only.
- The test split is evaluated after the equipment/signature/configuration is frozen.
- Dates are mutually exclusive and windows do not cross large data gaps.
- Identity and correlation graphs use the same encoder/readout for topology attribution.

## Reporting

Report Accuracy, Macro-Precision, Macro-Recall, Macro-F1, classwise metrics, confusion matrices, seed mean±std, split manifests and integrity audits. A high classification score on semi-physical faults must not be described as a field failure-detection rate.
