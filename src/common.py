"""Shared helpers: single-threaded BLAS pinning, input generation, validation, environment capture.

IMPORTANT: this module must be imported BEFORE numpy in every entry point, because it pins the
BLAS thread count (OPENBLAS_NUM_THREADS=1). Dask worker processes inherit the variable.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import platform
import sys
import numpy as np

SEED = 42
DTYPE = np.float64
RTOL = 1e-10   # float64 eps ~1.1e-16; a dot product of n<=4000 terms has relative error ~n*eps ~ 4e-13
ATOL = 1e-10   # (worst-case bound), so 1e-10 leaves >2 orders of magnitude of headroom yet catches real bugs.


def make_inputs(n, seed=SEED):
    """Shared generator: A then B drawn sequentially from numpy PCG64, U[0,1), float64."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n), dtype=DTYPE)
    B = rng.random((n, n), dtype=DTYPE)
    return A, B


def row_blocks(n, p):
    """Split n rows into p contiguous blocks (sizes differ by at most 1). Returns list of (lo, hi)."""
    edges = np.linspace(0, n, p + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(p) if edges[i + 1] > edges[i]]


def close(C, ref):
    return bool(np.allclose(C, ref, rtol=RTOL, atol=ATOL))


def max_rel_err(C, ref):
    return float(np.max(np.abs(C - ref)) / np.max(np.abs(ref)))


def naive_matmul(A, B):
    """Textbook triple loop in pure Python (validation oracle for small n only)."""
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def environment():
    import psutil
    import dask
    import distributed
    info = {
        "os": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
        "dask": dask.__version__, "distributed": distributed.__version__,
        "physical_cores": psutil.cpu_count(logical=False), "logical_processors": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "ram_available_gb_at_start": round(psutil.virtual_memory().available / 1e9, 2),
        "blas_threads_env": os.environ["OPENBLAS_NUM_THREADS"], "dtype": "float64", "seed": SEED,
        "generator": "numpy.random.default_rng(42) (PCG64), A then B, U[0,1)",
    }
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True).stdout.strip()
        info["cpu"] = out
    except Exception:
        info["cpu"] = platform.processor()
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg["Build Dependencies"]["blas"]
        info["blas"] = b["name"] + " " + str(b.get("version", ""))
    except Exception:
        info["blas"] = "unknown"
    return info
