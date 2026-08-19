"""Build the VHP 24-sensor A0 physical graph used by Level-C v04.

Important:
- Graph edges come only from this-unit DCS location / device-IO / direct thermal interfaces.
- No fault-signature information is used to define graph edges.
- Output is a binary undirected adjacency without self-loops; the training script handles self/neighbor paths separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_graph(tags: list[str]) -> np.ndarray:
    n = len(tags)
    if n != 24:
        raise ValueError(f"Expected 24 frozen VHP nodes, got {n}")
    tag2i = {t: i for i, t in enumerate(tags)}
    edges: list[tuple[int, int]] = []

    def add(a: str, b: str) -> None:
        if a not in tag2i or b not in tag2i:
            raise KeyError(f"Graph edge tag missing from frozen 24-node list: {a}, {b}")
        i, j = sorted((tag2i[a], tag2i[b]))
        if i != j and (i, j) not in edges:
            edges.append((i, j))

    # VHP blade-stage pressure progression.
    add("SYG1_10MAA50CP101", "SYG1_10MAA50CP102")
    add("SYG1_10MAA50CP102", "SYG1_10MAA50CP103")

    # Flow-path outlet to A/B exhaust vertical pipes.
    add("SYG1_10MAA50CP103", "SYG1_10LBC01CP101")
    add("SYG1_10MAA50CP103", "SYG1_10LBC02CP101")
    add("SYG1_10LBC01CP101", "SYG1_10LBC01CT101")
    add("SYG1_10LBC02CP101", "SYG1_10LBC02CT101")

    # Redundant VHP exhaust-steam temperature measurements belong to the outlet region.
    for t in ("SYG1_10MAA50CT121A", "SYG1_10MAA50CT122A", "SYG1_10MAA50CT123A"):
        add(t, "SYG1_10LBC01CT101")
        add(t, "SYG1_10LBC02CT101")

    # VHP outlet -> RH1 cold-reheat interface.
    add("SYG1_10LBC01CT101", "SYG1_10LBC20CT101")
    add("SYG1_10LBC02CT101", "SYG1_10LBC20CT101")
    add("SYG1_10LBC20CT101", "SYG1_10LBC20CT103")
    add("SYG1_10LBC20CT101", "SYG1_10LBC20CT104")

    # First-extraction branch: outlet -> before NRV -> after NRV -> No.1 HP heater inlet.
    add("SYG1_10LBQ10CP101", "SYG1_10LBQ10CT101A")
    for t in ("SYG1_10LBQ10CT102", "SYG1_10LBQ10CT103"):
        add("SYG1_10LBQ10CT101A", t)
    for pre in ("SYG1_10LBQ10CT102", "SYG1_10LBQ10CT103"):
        for post in ("SYG1_10LBQ10CT104", "SYG1_10LBQ10CT105"):
            add(pre, post)
    for post in ("SYG1_10LBQ10CT104", "SYG1_10LBQ10CT105"):
        add(post, "SYG1_10LBQ10CP102")
        add(post, "SYG1_10LBQ10CT106")
    add("SYG1_10LBQ10CP102", "SYG1_10LBQ10CT106")
    add("SYG1_10MAA50CP102", "SYG1_10LBQ10CP101")

    # Slow thermal-state branch.
    add("SYG1_10MAA50CT111A", "SYG1_10MAA50CT112A")
    add("SYG1_10MAA50CT111A", "SYG1_10MAA50CT131A")
    add("SYG1_10MAA50CT112A", "SYG1_10MAA50CT131A")
    add("SYG1_10MAA50CT131A", "SYG1_10MAA50CT123A")

    A = np.zeros((n, n), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="levelC_dataset_v03.npz")
    ap.add_argument("--out", default="kg_ordered_sensor_adjacency.csv")
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    tags = list(d["tags"].astype(str))
    A = build_graph(tags)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=tags, columns=tags).to_csv(out, encoding="utf-8-sig")
    print(f"nodes={len(tags)}, undirected_edges={int(A.sum()/2)}")
    print(out)


if __name__ == "__main__":
    main()
