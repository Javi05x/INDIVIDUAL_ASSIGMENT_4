"""Distributed dense matrix multiplication with Dask Distributed (1-D row-block partitioning).

Partitioning : A split into p contiguous row blocks A_i (p = number of workers); B replicated (broadcast) to every worker.
Placement    : task i computes C_i = A_i @ B and is PINNED to worker i (workers=[addr_i], allow_other_workers=False);
               A_i is scattered directly to worker i. One task per worker, 1 thread per worker, 1 BLAS thread.
Collection   : C_i blocks are gathered to the driver and copied into a preallocated C.
Phases (sequential, never overlapping by construction - each ends with a blocking wait):
   scatter  : driver -> workers transfer of A_i and B (blocking until data is resident)
   compute  : submit tasks until all futures finished
   gather   : workers -> driver transfer of C_i + copy into C
   total    : wall time from before scatter to after gather (end-to-end boundary)
"""
import common  # noqa: F401
import time
import numpy as np
import psutil
from dask.distributed import Client, LocalCluster, wait
from common import row_blocks


def _block_mm(a, b):
    t0 = time.perf_counter()
    c = a @ b
    return c, time.perf_counter() - t0


def _worker_peak_mem():
    return psutil.Process().memory_info().peak_wset  # Windows peak working set (bytes)


def _worker_rss():
    return psutil.Process().memory_info().rss


class DistributedMM:
    def __init__(self, n_workers):
        self.w = n_workers

    def __enter__(self):
        t0 = time.perf_counter()
        # processes=True: each worker is a separate OS process; memory_limit=0 disables dask's pause/kill logic
        self.cluster = LocalCluster(n_workers=self.w, threads_per_worker=1, processes=True,
                                    memory_limit=0, dashboard_address=None)
        self.client = Client(self.cluster)
        self.client.wait_for_workers(self.w)
        self.client.run(_worker_rss)  # force every worker to be fully up
        self.setup_s = time.perf_counter() - t0
        self.addrs = sorted(self.client.scheduler_info()["workers"])
        return self

    def __exit__(self, *exc):
        self.client.close()
        self.cluster.close()

    def worker_memory(self):
        """Peak working set per worker in bytes (Windows peak_wset, cumulative since worker start)."""
        return list(self.client.run(_worker_peak_mem).values())

    def multiply(self, A, B):
        n = A.shape[0]
        blocks = row_blocks(n, self.w)
        c = self.client
        t0 = time.perf_counter()
        # --- scatter ---
        bf = c.scatter(B, workers=self.addrs, broadcast=True, hash=False)
        af = [c.scatter(A[lo:hi], workers=[self.addrs[i]], hash=False) for i, (lo, hi) in enumerate(blocks)]
        wait([bf] + af)
        t1 = time.perf_counter()
        # --- compute ---
        futs = [c.submit(_block_mm, af[i], bf, workers=[self.addrs[i]], allow_other_workers=False, pure=False)
                for i in range(len(blocks))]
        wait(futs)
        t2 = time.perf_counter()
        # --- gather ---
        res = c.gather(futs)
        C = np.empty((n, B.shape[1]), dtype=A.dtype)
        for (lo, hi), (blk, _) in zip(blocks, res):
            C[lo:hi] = blk
        t3 = time.perf_counter()
        kern = [r[1] for r in res]
        for f in futs + af + [bf]:
            f.release()
        phases = {"scatter_s": t1 - t0, "compute_s": t2 - t1, "gather_s": t3 - t2, "total_s": t3 - t0,
                  "task_max_s": max(kern), "task_min_s": min(kern), "task_sum_s": sum(kern)}
        return C, phases
