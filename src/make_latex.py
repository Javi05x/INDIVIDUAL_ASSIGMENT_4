"""Generate the LaTeX source of the report from results/*.csv (all numbers come from the data)."""
import os
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "report", "Assignment04_BolivarGarciaIzquierdo_Report.tex")


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def tabular(header, rows, spec, size=r"\scriptsize"):
    out = [r"\begin{center}", size, r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


TEMPLATE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\graphicspath{{../figures/}}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}

\title{Distributed Matrix Multiplication:\\Scaling of a Dask Row-Block Implementation on a Single Machine}
\author{Javier Bolivar Garc\'ia-Izquierdo\\[2pt]\small BIG DATA -- Individual Assignment 4 (2026--2027)\\\small Grado en Ciencia e Ingenier\'ia de Datos, Universidad de Las Palmas de Gran Canaria}
\date{}

\begin{document}
\maketitle

\begin{abstract}
We study how a distributed dense matrix multiplication (Dask Distributed, 1-D row-block partitioning, broadcast of $B$) scales with the number of worker processes for $n=1000$--$4000$ (float64).
\textbf{Environment: local execution on one laptop (@PHYS@ physical cores, @LOG@ logical processors); all workers are processes on the same machine and no inter-machine network was involved.}
Against a validated sequential BLAS baseline, the best end-to-end speedup was @BESTS@$\times$ ($n=@BESTN@$, @BESTW@ workers, efficiency @BESTE@); the multiplication kernel alone reached @BKS@$\times$ ($n=@BKN@$, @BKW@ workers).
For $n=1000$ the distributed version is slower than sequential for every worker count.
Data movement (scatter of the $A$ blocks plus one replica of $B$ per worker) and framework overhead, rather than computation, limit scaling.
The conclusions apply to this machine and these sizes only.
\end{abstract}

\section{Introduction and objective}
\textbf{Research question.} How does the end-to-end time of a distributed dense matrix multiplication change when the number of workers grows at fixed problem size (strong scaling), and how much of it is computation versus data transfer and framework overhead?

\textbf{Hypotheses (stated before measuring).}
H1: computation scales almost linearly with workers because blocks are independent.
H2: dense multiplication performs $O(n^3)$ work on $O(n^2)$ data, so relative overhead shrinks as $n$ grows and larger $n$ scales better.
H3: broadcasting $B$ costs $p\cdot 8n^2$ bytes, so the scatter phase grows with the worker count $p$ and eventually dominates.

\section{Background and references}
Dense $C=AB$ costs $2n^3$ floating-point operations. With a row-block (1-D) decomposition, worker $i$ holds $A_i$ ($n/p$ rows) and a full copy of $B$ and computes $C_i=A_iB$ with no communication during the kernel~\cite{grama}. Total traffic is $(p+2)\cdot 8n^2$ bytes ($B$ to $p$ workers, $A$ once, $C$ once). 2-D schemes such as SUMMA~\cite{summa} reduce the replication of $B$ at the price of more communication steps; they are not implemented here. Dask Distributed provides a scheduler, worker processes and a futures API~\cite{dask}. Speedup is $S=T_\mathrm{seq}/T_p$ and efficiency is $S/p$~\cite{grama}. Dense kernels use NumPy~\cite{numpy} over OpenBLAS.

\section{Methodology}
\subsection{Implementation}
\emph{Partitioning:} $A$ is split into $p$ contiguous row blocks (sizes differ by at most one row); $B$ is replicated to every worker with \texttt{client.scatter(broadcast=True)}.
\emph{Placement:} block $i$ of $A$ is scattered to worker $i$, and task $i$ (\texttt{C\_i = A\_i @ B}, OpenBLAS \texttt{dgemm}) is pinned to worker $i$ (\texttt{allow\_other\_workers=False}): one task, one thread and one BLAS thread per worker process.
\emph{Collection:} the $C_i$ blocks are gathered into the driver and copied into a preallocated $C$.
Workers are OS processes created by \texttt{LocalCluster(processes=True, threads\_per\_worker=1)} (Distributed @DIST@). The per-worker memory limit is disabled (\texttt{memory\_limit=0}) to avoid spurious pause/kill logic; the global memory budget below is enforced instead.

\subsection{Baselines}
Materials of Assignments 1--3 were not available in the workspace, so nothing was reused; both baselines were written for this assignment and use the same kernel and inputs: (i) \textbf{sequential}: \texttt{A @ B} with one BLAS thread; (ii) \textbf{shared-memory parallel}: the same row-block decomposition over $k$ Python threads writing into a preallocated $C$ (NumPy releases the GIL in \texttt{dgemm}), one BLAS thread each. The parallel version isolates the effect of the decomposition without any data transfer.

\subsection{Environment (local execution)}
CPU: @CPU@; @PHYS@ physical cores / @LOG@ logical processors; RAM @RAM@ GB total (only @AVAIL@ GB available when the experiment started, which fixed the memory budget); @OS@; Python @PY@; NumPy @NP@ (@BLAS@); Dask/Distributed @DASK@. BLAS threads were pinned to @BT@ (\texttt{OPENBLAS/OMP/MKL\_NUM\_THREADS}) in the driver and inherited by the workers; Dask threads per worker $=1$.
\textbf{Machines: 1. Processes: one per worker. Workers: 1--16.} Running several workers locally measures framework and process overhead and loopback inter-process transfer; it does not measure inter-machine network performance or multi-machine memory capacity.
Local execution was used because no cluster was available. \emph{The instructor's permission for local execution is not confirmed in the course materials and must be verified before submission.}

\subsection{Inputs and correctness}
\texttt{numpy.random.default\_rng(42)} (PCG64), $A$ then $B$, uniform $[0,1)$, float64, generated once per $n$ and passed to every method (same logical inputs).
Before benchmarking (\texttt{validate.py}): known $2\times2$ product, identity, zero matrices, a pure-Python triple loop against NumPy ($n=40$), and threaded and distributed results against the reference for $n=1,2,7,257$ ($n<p$, $n$ not divisible by $p$, odd $n$), plus identity and zero through the cluster. All passed.
Every benchmark result was additionally compared with a reference product using $\mathrm{rtol}=\mathrm{atol}=10^{-10}$ (float64 $\varepsilon\approx1.1\cdot10^{-16}$; the worst-case error of an $n\le4000$ dot product is about $n\varepsilon\approx4\cdot10^{-13}$, leaving two orders of magnitude of headroom). All measured runs passed; the maximum relative error was @MAXERR@.

\subsection{Timing protocol}
\emph{End-to-end boundary (all methods):} $A$ and $B$ resident in driver memory $\rightarrow$ $C$ resident in driver memory. Included for Dask: scatter, task scheduling, computation, gather and copy into $C$; for the baselines: output allocation and the kernel. Excluded: input generation, reference computation, validation and \textbf{cluster start-up}, which is reported separately (median @SETUP1@~s for 1 worker to @SETUP16@~s for 16).
Timer: \texttt{time.perf\_counter} (monotonic). One warm-up run (discarded, not stored) and five measured repetitions per configuration; all raw values are in the CSV files; tables give the median and the interquartile range (IQR).
Dask phases are measured one after another, each ending with a blocking wait, so \textbf{scatter $+$ compute $+$ gather equals the end-to-end time by construction and the phases do not overlap}. ``Compute'' is the wall time from task submission until all futures finish and includes scheduler latency; ``kernel'' is the in-worker \texttt{perf\_counter} time of the multiplication of the slowest worker (the critical path). A fresh Python process and cluster is used for every $(n,p)$ pair so memory peaks are not contaminated.

\subsection{Budgets and memory metric}
Maximum 300~s per configuration (never reached; the longest run took @MAXT@~s). Memory budget 1.7~GB because the machine had about 1.85~GB free. Planning estimate: $4\cdot8n^2$ (driver: $A$, $B$, reference, $C$) $+\,p\cdot8n^2$ ($B$ replicas) $+\,2\cdot8n^2$ ($A$ and $C$ blocks) $+\,p\cdot80$~MB (idle worker interpreter).
Configurations above the budget were \textbf{not run} and no time is reported for them: @SKIPPED@.
The largest size meeting the limits is $n=4000$ with 4 workers; $n=4000$ with 8 workers was the first configuration attempted above the budget.
Memory metric: Windows \emph{peak working set} (\texttt{psutil.Process().memory\_info().peak\_wset}) per worker process and for the driver (which includes the inputs and the reference matrix). It is a resident-memory peak, not heap or array storage. Baseline peaks are cumulative over the process lifetime ($n$ ascending in one process) and are therefore upper bounds for the smaller sizes.

\section{Results}
@TABLE1@
\captionof{table}{Distributed (Dask) strong scaling. Seconds, median of 5 repetitions (1 warm-up discarded). Phases are sequential, so scatter$+$compute$+$gather$\approx$e2e (medians need not add exactly). Speedup is relative to the sequential median of Table~\ref{tab:base}. ``Transfer \%'' $=(\mathrm{e2e}-\mathrm{compute})/\mathrm{e2e}$. ``Kernel'' is the multiplication time of the slowest worker.}\label{tab:dist}

@TABLE2@
\captionof{table}{Baselines: kernel time in seconds, median of 5 repetitions; same boundary as the Dask end-to-end time without any transfer. Threaded runs use $k$ threads. ``Peak WS'' is the cumulative peak working set of the process.}\label{tab:base}

\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{fig1_scaling.png}
\caption{Left: strong-scaling speedup over the sequential baseline (solid: Dask end-to-end; dashed: shared-memory threads; dotted: ideal). Right: Dask end-to-end time; dotted lines are the sequential times. Log--log axes; configurations above the memory budget are absent.}\label{fig:scaling}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{fig2_phases.png}
\caption{Dask end-to-end time split into its three sequential phases (medians). Dashed: sequential baseline for the same $n$.}\label{fig:phases}
\end{figure}

@TABLE3@
\captionof{table}{Data volume, effective transfer rate and memory. Bytes moved $=(p+2)\cdot8n^2$ (computed from the algorithm, not measured on a network). Throughput $=$ bytes\,/\,phase time and therefore includes serialisation and scheduling; it is loopback inter-process throughput, not network bandwidth. Peak working set: maximum over workers.}\label{tab:mem}

\begin{figure}[ht]
\centering
\includegraphics[width=0.55\textwidth]{fig3_memory.png}
\caption{Peak working set per worker process versus number of workers (maximum over workers).}\label{fig:mem}
\end{figure}

\section{Discussion}
\textbf{Computation (H1: partly supported).} The in-worker kernel scales well up to 4 workers (kernel efficiency @KE34@ at $n=3000$; @KE44@ at $n=4000$) and degrades at 8 (@KE38@ at $n=3000$). The shared-memory threaded baseline shows the same pattern with no transfers at all (speedup about 4 at 8 threads, no gain at 16), so the ceiling is a property of this machine and kernel, not of Dask. Plausible causes, not isolated here because no hardware counters were collected: memory bandwidth, the 8 physical cores being shared with SMT siblings, and power or thermal limits of a laptop CPU. With one worker the kernel time equals the sequential time (ratio @KS31@ at $n=3000$), so the worker process itself adds no kernel penalty.

\textbf{End-to-end (H2 and H3: supported).} End-to-end speedup is far below kernel speedup. With one worker the distributed run is slower than sequential (@SE31@$\times$ at $n=3000$) because $A$ and $B$ are copied to the worker and $C$ back (@OV1@\% of the time at $n=1000$ down to @OV4@\% at $n=4000$), consistent with the $O(n^3)/O(n^2)$ argument. Scatter grows with the worker count because the replicas of $B$ amount to $p\cdot8n^2$ bytes ($n=3000$: @SC31@~s with 1 worker, @SC38@~s with 8) while compute shrinks, so beyond 4 workers the end-to-end time stops improving ($n=3000$: @T34@~s at 4 workers, @T38@~s at 8). For $n=1000$ the problem (about 35~ms sequentially) is too small: every Dask configuration is slower than sequential and 16 workers (more than the 8 physical cores) take @T1016@~s. Dask beats sequential only for $n\ge2000$ with 2--4 workers (best @BESTS@$\times$) and never matches the shared-memory baseline, which is the cost of crossing process boundaries.

\textbf{Memory.} The per-worker peak working set falls with the number of workers ($n=3000$: @WM31@~MB $\rightarrow$ @WM38@~MB) but far less than $1/p$, because every worker keeps a full replica of $B$ (72~MB at $n=3000$) plus the interpreter and BLAS baseline; the sum over workers grows with $p$. Row-block partitioning with broadcast therefore does not remove the single-node memory requirement for $B$ and cannot handle matrices whose $B$ does not fit on one worker. Inputs larger than one machine's memory were not attempted (an extension requiring infrastructure).

\textbf{Limitations.}
(1) Single machine: transfer cost is inter-process communication over loopback and cannot be extrapolated to a real network; network cost was not isolated and no network measurement was invented.
(2) Free RAM ($\sim$1.85~GB of 16~GB) capped the size at $n=4000$ and excluded 8 and 16 workers at the largest sizes; scaling for $n\ge4000$ with 8 workers is untested.
(3) Five repetitions, one machine session, other processes running; IQRs are small ($\le0.08$~s) except for oversubscribed configurations ($n=1000$, 16 workers: IQR 0.12~s on a 0.21~s median).
(4) Weak scaling was not studied.
(5) Dask internals (serialisation, scheduler latency) are not separated: ``compute'' wall time minus ``kernel'' is an upper bound of scheduling overhead.
(6) Peak working set is Windows-specific and not comparable to heap or array size.
(7) The scatter phase combines the replication of $B$ and the distribution of $A$.
(8) 2-D partitioning, which would reduce replicated traffic, was not implemented.

\section{Conclusions}
Within the tested scope (one laptop, float64, $n=1000$--$4000$, up to 8 workers for $n\le3000$ and 4 for $n=4000$), the multiplication kernel parallelises well across worker processes up to about 4--8 workers, but a row-block Dask implementation that broadcasts $B$ yields at most @BESTS@$\times$ end-to-end over sequential BLAS because transfer and framework overhead grow with the number of workers and are significant at these sizes. Whether the same trade-off holds on a real network, for larger $n$ or with a 2-D decomposition remains unresolved and requires a multi-machine experiment.

\section{References and materials}
\begin{thebibliography}{9}\small
\bibitem{grama} A. Grama, A. Gupta, G. Karypis and V. Kumar, \emph{Introduction to Parallel Computing}, 2nd ed., Addison-Wesley, 2003.
\bibitem{summa} R. A. van de Geijn and J. Watts, ``SUMMA: Scalable universal matrix multiplication algorithm,'' \emph{Concurrency: Practice and Experience}, vol. 9, no. 4, pp. 255--274, 1997.
\bibitem{dask} Dask Development Team, ``Dask Distributed documentation,'' \url{https://distributed.dask.org} (accessed Nov. 2026).
\bibitem{numpy} C. R. Harris et al., ``Array programming with NumPy,'' \emph{Nature}, vol. 585, pp. 357--362, 2020.
\end{thebibliography}

\textbf{Materials.} Folder \texttt{assignment4/} (to be published in the course GitHub repository): \texttt{src/} (code), \texttt{results/*\_raw.csv} (all raw measurements), \texttt{results/summary\_*.csv}, \texttt{results/environment.json}, \texttt{results/skipped.csv}, \texttt{figures/}, \texttt{README.md} (exact commands), \texttt{requirements.txt}. Reproduce: \texttt{python src/run\_all.py \&\& python src/analyze.py \&\& python src/make\_latex.py}.

\textbf{Generative-AI disclosure.} Claude (Anthropic; Sonnet 5.5, with Haiku used to read the assignment PDF) was used to plan the experiment, write the code and draft this report. All numbers in tables and figures are generated by the included scripts from the raw CSV files; the author is responsible for understanding the code, verifying the measurements and references, and defending the conclusions.

\end{document}
"""


def main():
    env = json.load(open(os.path.join(RES, "environment.json")))
    raw = pd.read_csv(os.path.join(RES, "distributed_raw.csv"))
    d = pd.read_csv(os.path.join(RES, "summary_distributed.csv"))
    bs = pd.read_csv(os.path.join(RES, "summary_baselines.csv"))
    ps = pd.read_csv(os.path.join(RES, "summary_parallel.csv"))
    sk = pd.read_csv(os.path.join(RES, "skipped.csv"))
    seq = bs[bs.method == "sequential"].set_index("n")
    t = d.set_index(["n", "workers"])
    best = d.loc[d.speedup_e2e.idxmax()]
    bk = d.loc[d.speedup_kernel.idxmax()]

    rows = [[int(r.n), int(r.workers), f"{r.total_med:.3f}", f"{r.total_iqr:.3f}", f"{r.scatter_med:.3f}", f"{r.compute_med:.3f}",
             f"{r.gather_med:.3f}", f"{r.task_max_med:.3f}", f"{r.speedup_e2e:.2f}", f"{r.efficiency_e2e:.2f}",
             f"{r.speedup_kernel:.2f}", f"{r.overhead_pct:.0f}"] for r in d.itertuples()]
    table1 = tabular(["$n$", "$p$", "e2e", "IQR", "scatter", "compute", "gather", "kernel", "S e2e", "E e2e", "S kernel", "transfer \\%"],
                     rows, "rrrrrrrrrrrr", r"\scriptsize")
    rows = []
    for n in sorted(seq.index):
        thr = {k: bs[(bs.method == "parallel_threads") & (bs.n == n) & (bs.workers == k)].med.iloc[0] for k in (2, 4, 8, 16)}
        sp8 = ps[(ps.n == n) & (ps.workers == 8)].speedup.iloc[0]
        rows.append([n, f"{seq.loc[n, 'med']:.4f}", f"{seq.loc[n, 'iqr']:.4f}"] + [f"{thr[k]:.4f}" for k in (2, 4, 8, 16)] +
                    [f"{sp8:.2f}", f"{seq.loc[n, 'peak_mb']:.0f}"])
    table2 = tabular(["$n$", "sequential", "IQR", "$k$=2", "$k$=4", "$k$=8", "$k$=16", "S ($k$=8)", "Peak WS (MB)"], rows, "rrrrrrrrr")
    rows = [[int(r.n), int(r.workers), f"{r.bytes_moved_mb:.0f}", f"{r.scatter_MBps:.0f}", f"{r.gather_MBps:.0f}",
             f"{r.worker_peak_mb:.0f}", f"{r.driver_peak_mb:.0f}"] for r in d.itertuples()]
    table3 = tabular(["$n$", "$p$", "Moved (MB)", "Scatter MB/s", "Gather MB/s", "Worker peak WS (MB)", "Driver peak WS (MB)"], rows, "rrrrrrr")

    setup = d.groupby("workers").setup_s.median()
    skipped = "; ".join(f"$n={int(r.n)}$, $p={int(r.workers)}$ (est.\\ {r.est_memory_gb}~GB)" for r in sk.itertuples())
    rep = {
        "PHYS": env["physical_cores"], "LOG": env["logical_processors"],
        "BESTS": f"{best.speedup_e2e:.2f}", "BESTN": int(best.n), "BESTW": int(best.workers), "BESTE": f"{best.efficiency_e2e:.2f}",
        "BKS": f"{bk.speedup_kernel:.2f}", "BKN": int(bk.n), "BKW": int(bk.workers),
        "DIST": env["distributed"], "CPU": esc(env["cpu"]), "RAM": env["ram_total_gb"], "AVAIL": env["ram_available_gb_at_start"],
        "OS": esc(env["os"]), "PY": env["python"], "NP": env["numpy"], "BLAS": esc(env["blas"]), "DASK": env["dask"],
        "BT": env["blas_threads_env"], "MAXERR": "$" + f"{d.max_rel_err.max():.1e}".replace("e-", r"\times10^{-") + "}$",
        "SETUP1": f"{setup.loc[1]:.1f}", "SETUP16": f"{setup.loc[16]:.1f}", "MAXT": f"{raw.total_s.max():.1f}", "SKIPPED": skipped,
        "KE34": f"{t.loc[(3000, 4), 'efficiency_kernel']:.2f}", "KE44": f"{t.loc[(4000, 4), 'efficiency_kernel']:.2f}",
        "KE38": f"{t.loc[(3000, 8), 'efficiency_kernel']:.2f}", "KS31": f"{t.loc[(3000, 1), 'speedup_kernel']:.2f}",
        "SE31": f"{t.loc[(3000, 1), 'speedup_e2e']:.2f}", "OV1": f"{t.loc[(1000, 1), 'overhead_pct']:.0f}",
        "OV4": f"{t.loc[(4000, 1), 'overhead_pct']:.0f}", "SC31": f"{t.loc[(3000, 1), 'scatter_med']:.2f}",
        "SC38": f"{t.loc[(3000, 8), 'scatter_med']:.2f}", "T34": f"{t.loc[(3000, 4), 'total_med']:.2f}",
        "T38": f"{t.loc[(3000, 8), 'total_med']:.2f}", "T1016": f"{t.loc[(1000, 16), 'total_med']:.2f}",
        "WM31": f"{t.loc[(3000, 1), 'worker_peak_mb']:.0f}", "WM38": f"{t.loc[(3000, 8), 'worker_peak_mb']:.0f}",
        "TABLE1": table1, "TABLE2": table2, "TABLE3": table3,
    }
    tex = TEMPLATE
    for k, v in rep.items():
        tex = tex.replace(f"@{k}@", str(v))
    import re
    left = re.findall(r"@[A-Z0-9]+@", tex)
    assert not left, f"unreplaced placeholders: {left}"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print("latex ->", OUT)


if __name__ == "__main__":
    main()
