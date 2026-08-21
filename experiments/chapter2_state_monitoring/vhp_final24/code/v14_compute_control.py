from __future__ import annotations

"""Shared resource guard for VHP training jobs.

Call ``configure_compute`` before constructing models or large NumPy jobs.
The default keeps four CPU threads for preprocessing while the RTX GPU handles
training.  Override explicitly with ``--cpu-threads`` or ``VHP_CPU_THREADS``.
"""

import os


DEFAULT_CPU_THREADS = max(1, min(4, int(os.environ.get("VHP_CPU_THREADS", "4"))))
_RUNTIME_LIMITER = None


def configure_compute(cpu_threads: int = DEFAULT_CPU_THREADS) -> dict[str, int | str]:
    global _RUNTIME_LIMITER
    cpu_threads = max(1, int(cpu_threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(cpu_threads)

    # Import after environment variables so OpenMP/BLAS see the requested cap.
    import torch

    # Training modules may already have imported NumPy/SciPy before calling
    # this function.  Environment variables alone are then too late for the
    # already-loaded MKL/OpenBLAS runtime, so also apply a live threadpool cap.
    try:
        from threadpoolctl import threadpool_limits

        _RUNTIME_LIMITER = threadpool_limits(limits=cpu_threads)
    except ImportError:
        _RUNTIME_LIMITER = None

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    status = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "dataloader_workers": 0,
    }
    print("[compute] " + " ".join(f"{key}={value}" for key, value in status.items()), flush=True)
    return status
