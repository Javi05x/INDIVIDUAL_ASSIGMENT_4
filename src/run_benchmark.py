"""Benchmark driver.

  python run_benchmark.py baselines --n 1000 2000 --k 1 2 4 8 16
  python run_benchmark.py distributed --n 2000 --w 4          (one fresh process per (n, w) => clean memory peaks)

Protocol: 1 warm-up run (discarded, not in CSV) + REPS measured repetitions; every measurement is appended
to the CSV (raw data kept). Result of every run (warm-up excluded) is checked against a reference
computed once, outside the timed region.
Timer boundary (all methods): A and B resident in driver memory  ->  C resident in driver memory.
Input generation, reference computation, validation and cluster start-up are OUTSIDE the boundary
(start-up is reported separately as setup_s).
"""
import common  # noqa: F401  must be first
import argparse
import csv
import os
import time
import numpy as np
import psutil
from common import make_inputs, close, max_rel_err

REPS = 5
WARMUP = 1
MAX_RUN_S = 300             # declared per-configuration time budget
MEM_BUDGET = 1.7e9          # declared memory budget (bytes): free RAM on the test machine was ~2 GB
WORKER_BASE = 80e6          # estimated resident size of an idle Dask worker process (bytes)
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def estimated_memory(n, w):
    """Planning estimate of total RAM: driver (A,B,ref,C) + B replica per worker + A blocks + C blocks + interpreters."""
    m = 8 * n * n
    return 4 * m + w * m + 2 * m + w * WORKER_BASE


def append(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if new:
            wr.writeheader()
        wr.writerow(row)


def peak_mb():
    return psutil.Process().memory_info().peak_wset / 1e6


def run_baselines(ns, ks):
    from baselines import sequential, parallel_threads
    path = os.path.join(RES, "baselines_raw.csv")
    fields = ["method", "n", "workers", "rep", "total_s", "max_rel_err", "ok", "process_peak_mb"]
    for n in ns:
        A, B = make_inputs(n)
        ref = sequential(A, B)
        for method, k in [("sequential", 1)] + [("parallel_threads", k) for k in ks]:
            f = (lambda: sequential(A, B)) if method == "sequential" else (lambda k=k: parallel_threads(A, B, k))
            for _ in range(WARMUP):
                f()
            for rep in range(1, REPS + 1):
                t0 = time.perf_counter()
                C = f()
                dt = time.perf_counter() - t0
                append(path, fields, {"method": method, "n": n, "workers": k, "rep": rep, "total_s": dt,
                                      "max_rel_err": max_rel_err(C, ref), "ok": close(C, ref),
                                      "process_peak_mb": peak_mb()})
            print(f"{method} n={n} k={k} done", flush=True)
        del A, B, ref


def run_distributed(n, w):
    from distributed_mm import DistributedMM
    path = os.path.join(RES, "distributed_raw.csv")
    fields = ["method", "n", "workers", "rep", "total_s", "scatter_s", "compute_s", "gather_s", "task_max_s",
              "task_min_s", "task_sum_s", "setup_s", "max_rel_err", "ok", "driver_peak_mb",
              "worker_peak_mb_max", "worker_peak_mb_sum", "bytes_moved"]
    if estimated_memory(n, w) > MEM_BUDGET:
        append(os.path.join(RES, "skipped.csv"), ["n", "workers", "reason", "est_memory_gb"],
               {"n": n, "workers": w, "reason": "estimated memory exceeds declared budget",
                "est_memory_gb": round(estimated_memory(n, w) / 1e9, 2)})
        print(f"SKIP n={n} w={w} (est {estimated_memory(n, w)/1e9:.2f} GB > budget)")
        return
    A, B = make_inputs(n)
    ref = A @ B
    with DistributedMM(w) as dm:
        for _ in range(WARMUP):
            dm.multiply(A, B)
        rows = []
        for rep in range(1, REPS + 1):
            C, ph = dm.multiply(A, B)
            rows.append((rep, C, ph))
            if ph["total_s"] > MAX_RUN_S:
                print("run exceeded time budget; stopping repetitions early")
                break
        wm = dm.worker_memory()
        for rep, C, ph in rows:
            append(path, fields, {"method": "dask_row_blocks", "n": n, "workers": w, "rep": rep,
                                  "total_s": ph["total_s"], "scatter_s": ph["scatter_s"],
                                  "compute_s": ph["compute_s"], "gather_s": ph["gather_s"],
                                  "task_max_s": ph["task_max_s"], "task_min_s": ph["task_min_s"],
                                  "task_sum_s": ph["task_sum_s"], "setup_s": dm.setup_s,
                                  "max_rel_err": max_rel_err(C, ref), "ok": close(C, ref),
                                  "driver_peak_mb": peak_mb(), "worker_peak_mb_max": max(wm) / 1e6,
                                  "worker_peak_mb_sum": sum(wm) / 1e6, "bytes_moved": (w + 2) * 8 * n * n})
    print(f"distributed n={n} w={w} done ({len(rows)} reps)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baselines", "distributed", "env"])
    ap.add_argument("--n", type=int, nargs="+", default=[1000])
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--w", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.mode == "baselines":
        run_baselines(a.n, a.k)
    elif a.mode == "distributed":
        run_distributed(a.n[0], a.w)
    else:
        import json
        from common import environment
        with open(os.path.join(RES, "environment.json"), "w") as f:
            json.dump(environment(), f, indent=2)
