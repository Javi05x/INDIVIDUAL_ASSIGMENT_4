"""Baselines: validated sequential version and shared-memory row-block parallel version.

Both use the same logical inputs and the same dense float64 kernel (BLAS dgemm, 1 BLAS thread)
as the distributed version. Timer boundary = A, B resident in memory -> C resident in memory
(input generation excluded; output allocation included).
"""
import common  # noqa: F401
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from common import row_blocks


def sequential(A, B):
    return A @ B


def parallel_threads(A, B, k):
    """Row-block decomposition over k threads (numpy releases the GIL inside dgemm)."""
    n = A.shape[0]
    C = np.empty((n, B.shape[1]), dtype=A.dtype)

    def work(lohi):
        lo, hi = lohi
        np.matmul(A[lo:hi], B, out=C[lo:hi])

    with ThreadPoolExecutor(k) as ex:
        list(ex.map(work, row_blocks(n, k)))
    return C
