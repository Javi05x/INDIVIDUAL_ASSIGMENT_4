"""Build summary tables (median, IQR) and figures from the raw CSVs."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)


def main():
    d = pd.read_csv(os.path.join(RES, "distributed_raw.csv"))
    b = pd.read_csv(os.path.join(RES, "baselines_raw.csv"))
    assert d["ok"].all() and b["ok"].all(), "a result failed validation"

    ds = d.groupby(["n", "workers"]).agg(
        reps=("rep", "count"), total_med=("total_s", "median"), total_iqr=("total_s", iqr),
        scatter_med=("scatter_s", "median"), compute_med=("compute_s", "median"), gather_med=("gather_s", "median"),
        task_max_med=("task_max_s", "median"), task_min_med=("task_min_s", "median"),
        setup_s=("setup_s", "median"), worker_peak_mb=("worker_peak_mb_max", "max"),
        driver_peak_mb=("driver_peak_mb", "max"), bytes_moved_mb=("bytes_moved", lambda x: x.iloc[0] / 1e6),
        max_rel_err=("max_rel_err", "max")).reset_index()
    bs = b.groupby(["method", "n", "workers"]).agg(reps=("rep", "count"), med=("total_s", "median"),
                                                    iqr=("total_s", iqr), peak_mb=("process_peak_mb", "max")).reset_index()
    seq = bs[bs.method == "sequential"].set_index("n")["med"]
    ds["seq_med"] = ds["n"].map(seq)
    ds["speedup_e2e"] = ds["seq_med"] / ds["total_med"]
    ds["efficiency_e2e"] = ds["speedup_e2e"] / ds["workers"]
    ds["speedup_kernel"] = ds["seq_med"] / ds["task_max_med"]
    ds["efficiency_kernel"] = ds["speedup_kernel"] / ds["workers"]
    ds["overhead_pct"] = 100 * (ds["total_med"] - ds["compute_med"]) / ds["total_med"]
    # throughput of the scatter phase: bytes sent = (w+1)*8n^2 (B to each worker + all of A)
    ds["scatter_MBps"] = (ds["workers"] + 1) * 8 * ds["n"] ** 2 / 1e6 / ds["scatter_med"]
    ds["gather_MBps"] = 8 * ds["n"] ** 2 / 1e6 / ds["gather_med"]
    ps = bs[bs.method == "parallel_threads"].merge(seq.rename("seq_med"), on="n")
    ps["speedup"] = ps["seq_med"] / ps["med"]
    ps["efficiency"] = ps["speedup"] / ps["workers"]
    ds.to_csv(os.path.join(RES, "summary_distributed.csv"), index=False)
    bs.to_csv(os.path.join(RES, "summary_baselines.csv"), index=False)
    ps.to_csv(os.path.join(RES, "summary_parallel.csv"), index=False)

    ns = sorted(ds.n.unique())
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(ns)))

    # Fig 1: strong scaling speedup (end-to-end distributed vs threads vs ideal)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, n in zip(cols, ns):
        x = ds[ds.n == n]
        ax[0].plot(x.workers, x.speedup_e2e, "o-", color=c, label=f"Dask end-to-end n={n}")
        y = ps[ps.n == n]
        ax[0].plot(y.workers, y.speedup, "s--", color=c, alpha=.6, label=f"threads n={n}")
    ws = sorted(ds.workers.unique())
    ax[0].plot(ws, ws, "k:", label="ideal")
    ax[0].set(xscale="log", yscale="log", xlabel="workers (processes) / threads", ylabel="speedup vs sequential (x)",
              title="Strong scaling: speedup (median of 5)")
    ax[0].set_xticks(ws, [str(w) for w in ws]); ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=.3)
    for c, n in zip(cols, ns):
        x = ds[ds.n == n]
        ax[1].plot(x.workers, x.total_med, "o-", color=c, label=f"n={n}")
        ax[1].axhline(seq[n], color=c, ls=":", lw=1)
    ax[1].set(xscale="log", yscale="log", xlabel="workers (processes)", ylabel="end-to-end time (s)",
              title="End-to-end time (dotted = sequential)")
    ax[1].set_xticks(ws, [str(w) for w in ws]); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_scaling.png"), dpi=160); plt.close(fig)

    # Fig 2: phase breakdown (sequential phases, additive) for each n
    fig, axs = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 3.8), sharey=False)
    for ax, n in zip(np.atleast_1d(axs), ns):
        x = ds[ds.n == n].sort_values("workers")
        idx = np.arange(len(x))
        ax.bar(idx, x.scatter_med, label="scatter (A blocks + B broadcast)", color="#d95f02")
        ax.bar(idx, x.compute_med, bottom=x.scatter_med, label="compute (wall)", color="#1b9e77")
        ax.bar(idx, x.gather_med, bottom=x.scatter_med + x.compute_med, label="gather (C blocks)", color="#7570b3")
        ax.plot(idx, [seq[n]] * len(idx), "k--", lw=1, label="sequential")
        ax.set_xticks(idx, x.workers.astype(str)); ax.set(xlabel="workers", ylabel="time (s)", title=f"n={n}")
    np.atleast_1d(axs)[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_phases.png"), dpi=160); plt.close(fig)

    # Fig 3: memory per worker + estimated total
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for c, n in zip(cols, ns):
        x = ds[ds.n == n]
        ax.plot(x.workers, x.worker_peak_mb, "o-", color=c, label=f"n={n}")
    ax.set(xscale="log", xlabel="workers", ylabel="peak working set per worker (MB)", title="Memory per worker (max over workers)")
    ax.set_xticks(ws, [str(w) for w in ws]); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_memory.png"), dpi=160); plt.close(fig)
    print("analysis done")


if __name__ == "__main__":
    main()
