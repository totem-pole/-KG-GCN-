from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;out={}
for device in ["H2","H3","H4","L6","L9"]:
 droot=ROOT/"expanded_fault_data"/device;d=np.load(droot/f"{device}_fault_dataset.npz",allow_pickle=True);man={s:pd.read_csv(droot/f"{s}_manifest.csv") for s in ["train","val","test"]};dates={s:set(x.date.astype(str)) for s,x in man.items()};hs={s:{hashlib.sha256(x.tobytes()).hexdigest() for x in d[f"X_{s}"]} for s in ["train","val","test"]}
 out[device]={"nodes":int(d["X_train"].shape[1]),"classes":list(d["class_names"].astype(str)),"samples":{s:int(len(d[f"y_{s}"])) for s in ["train","val","test"]},"class_counts":{s:np.bincount(d[f"y_{s}"]).tolist() for s in ["train","val","test"]},"date_overlap_train_val":sorted(dates["train"]&dates["val"]),"date_overlap_train_test":sorted(dates["train"]&dates["test"]),"date_overlap_val_test":sorted(dates["val"]&dates["test"]),"exact_duplicate_train_val":len(hs["train"]&hs["val"]),"exact_duplicate_train_test":len(hs["train"]&hs["test"]),"exact_duplicate_val_test":len(hs["val"]&hs["test"])}
(ROOT/"expanded_deliverables").mkdir(parents=True,exist_ok=True);(ROOT/"expanded_deliverables"/"dataset_leakage_audit.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(out,ensure_ascii=False,indent=2))
