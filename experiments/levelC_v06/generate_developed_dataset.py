from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path('experiments_local/levelC_experiment_v03')
OUT = Path('experiments_local/levelC_experiment_v06')
OUT.mkdir(parents=True, exist_ok=True)

# Required local input: levelC_dataset_v03.npz + train/val/test_manifest.csv
# v06 preserves the same healthy windows, split, fault signature and temporal profile.
# Only the semi-physical fault severity is remapped to 1.5--2.5 sigma.

d = np.load(SRC / 'levelC_dataset_v03.npz', allow_pickle=True)
rng = np.random.default_rng(20260820)
save = {}
summary = []

for split in ['train', 'val', 'test']:
    X = d[f'X_{split}'].copy()
    y = d[f'y_{split}'].copy()
    manifest = pd.read_csv(SRC / f'{split}_manifest.csv')
    Xnew = X.copy()
    mnew = manifest.copy()

    # v03 samples are stored as groups of four:
    # [F0 healthy, F1, F2, F3], sharing the same real healthy residual background.
    for i in range(0, len(X), 4):
        base = X[i].copy()
        for j in (1, 2, 3):
            k = i + j
            old_severity = float(manifest.loc[k, 'severity'])
            new_severity = float(rng.uniform(1.5, 2.5))
            # Preserve original spatial signature, onset and temporal profile.
            delta = X[k] - base
            Xnew[k] = base + delta * (new_severity / max(old_severity, 1e-6))
            mnew.loc[k, 'severity'] = new_severity

    save[f'X_{split}'] = Xnew.astype(np.float32)
    save[f'y_{split}'] = y
    mnew.to_csv(OUT / f'{split}_manifest_developed_v06.csv', index=False, encoding='utf-8-sig')

    faults = mnew[mnew['fault'] != 'F0_normal']
    summary.append({
        'split': split,
        'samples': len(Xnew),
        'severity_min': faults['severity'].min(),
        'severity_mean': faults['severity'].mean(),
        'severity_max': faults['severity'].max(),
    })

save['tags'] = d['tags']
np.savez_compressed(OUT / 'levelC_dataset_developed_v06.npz', **save)
pd.DataFrame(summary).to_csv(OUT / 'dataset_developed_v06_summary.csv', index=False, encoding='utf-8-sig')
print(pd.DataFrame(summary))
