"""Generate the PDF report from results/*.csv and figures/*.png (reportlab)."""
import os
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(HERE, "..", "report", "Assignment04_BolivarGarciaIzquierdo_Report.pdf")

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=12.2, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading2"], fontSize=12.5, spaceBefore=9, spaceAfter=4)
T = ParagraphStyle("T", parent=ss["Title"], fontSize=17, leading=21)
C = ParagraphStyle("C", parent=ss["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"), spaceAfter=8)
SM = ParagraphStyle("SM", parent=B, fontSize=7.8, leading=10)


def P(t, s=B):
    return Paragraph(t, s)


def table(data, widths=None, fs=7.6):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(os.path.join(FIG, name)).size
    return Image(os.path.join(FIG, name), width=w, height=w * ih / iw)


def main():
    env = json.load(open(os.path.join(RES, "environment.json")))
    d = pd.read_csv(os.path.join(RES, "summary_distributed.csv"))
    bs = pd.read_csv(os.path.join(RES, "summary_baselines.csv"))
    ps = pd.read_csv(os.path.join(RES, "summary_parallel.csv"))
    sk = pd.read_csv(os.path.join(RES, "skipped.csv"))
    seq = bs[bs.method == "sequential"].set_index("n")

    s = []
    s += [P("Distributed Matrix Multiplication: Scaling of a Dask Row-Block Implementation on a Single Machine", T),
          P("<b>Javier Bolivar Garc&iacute;a-Izquierdo</b> &mdash; BIG DATA, Individual Assignment 4 (2026-2027) &mdash; "
            "Grado en Ciencia e Ingenier&iacute;a de Datos, ULPGC", ParagraphStyle("a", parent=B, alignment=1)),
          Spacer(1, 6)]

    best = d.loc[d.speedup_e2e.idxmax()]
    bk = d.loc[d.speedup_kernel.idxmax()]
    s += [P("Abstract", H1),
          P(f"We study how a distributed dense matrix multiplication (Dask Distributed, 1-D row-block partitioning, B broadcast) scales "
            f"with the number of worker processes for n = 1000&ndash;4000 (float64). <b>Environment: local, one laptop (8 physical cores, "
            f"16 logical), all workers are processes on the same machine; no network between machines was involved.</b> "
            f"Against a validated sequential BLAS baseline, the best end-to-end speedup was {best.speedup_e2e:.2f}x "
            f"(n={int(best.n)}, {int(best.workers)} workers, efficiency {best.efficiency_e2e:.2f}); the compute kernel alone reached "
            f"{bk.speedup_kernel:.2f}x (n={int(bk.n)}, {int(bk.workers)} workers). For n=1000 the distributed version is slower than sequential for every "
            f"worker count. Data movement (scatter of A blocks plus a B replica per worker) and framework overhead, not computation, "
            f"limit scaling. Conclusions apply to this machine and these sizes only.")]

    s += [P("1. Introduction and objective", H1),
          P("<b>Research question.</b> How does the end-to-end time of a distributed dense matrix multiplication change when the "
            "number of workers grows at fixed problem size (strong scaling), and how much of it is computation versus data "
            "transfer and framework overhead? "
            "<b>Hypotheses (made before measuring).</b> H1: Computation scales almost linearly with workers because blocks are independent. "
            "H2: Because dense multiplication does O(n<super>3</super>) work on O(n<super>2</super>) data, overhead shrinks relatively as n grows, so larger n scales better. "
            "H3: Broadcasting B costs w&middot;8n<super>2</super> bytes, so the scatter phase grows with w and eventually dominates."),
          P("2. Background", H1),
          P("Dense C=AB costs 2n<super>3</super> flops and each element requires n multiply-adds. With a row-block (1-D) decomposition, "
            "worker i owns A<sub>i</sub> (n/p rows) and a full copy of B and computes C<sub>i</sub>=A<sub>i</sub>B without communication during "
            "the kernel [1]. Total traffic is (p+2)&middot;8n<super>2</super> bytes (B to p workers, A once, C once). 2-D/SUMMA schemes reduce B replication "
            "but need more communication steps [2]; they are not implemented here. Dask Distributed provides a scheduler and worker processes "
            "with a client API for futures [3]. Speedup is S=T<sub>seq</sub>/T<sub>p</sub> and efficiency S/p [1]."),
          P("3. Methodology", H1)]

    s += [P("<b>3.1 Implementation.</b> <i>Partitioning:</i> A is split into p contiguous row blocks (sizes differ by at most one row); "
            "B is replicated to every worker with <font face='Courier'>client.scatter(broadcast=True)</font>. <i>Placement:</i> block i of A is "
            "scattered to worker i and task i (<font face='Courier'>C_i = A_i @ B</font>, NumPy/OpenBLAS dgemm) is pinned to worker i "
            "(<font face='Courier'>allow_other_workers=False</font>): one task, one thread and one BLAS thread per worker process. "
            "<i>Collection:</i> C<sub>i</sub> blocks are gathered into the driver and copied into a preallocated C. "
            "Workers are OS processes created by <font face='Courier'>LocalCluster(processes=True, threads_per_worker=1)</font> "
            f"(Dask {env['distributed']}); worker memory limit disabled (<font face='Courier'>memory_limit=0</font>) to avoid "
            "spurious pause/kill logic under our declared global budget.")]
    s += [P("<b>3.2 Baselines (reused: none &mdash; written for this assignment).</b> The materials of Assignments 1&ndash;3 were not available in this workspace, "
            "so both baselines were written here and use the <i>same</i> kernel and inputs: (i) <b>sequential</b>: <font face='Courier'>A @ B</font>, one BLAS thread; "
            "(ii) <b>parallel (shared memory)</b>: same row-block decomposition over k Python threads writing into a preallocated C (NumPy releases the GIL in dgemm), 1 BLAS thread each. "
            "The parallel version isolates the benefit of the decomposition without any data transfer.")]
    s += [P(f"<b>3.3 Environment (local execution &mdash; label).</b> CPU {env['cpu']}, {env['physical_cores']} physical cores / {env['logical_processors']} logical processors, "
            f"RAM {env['ram_total_gb']} GB total (only {env['ram_available_gb_at_start']} GB available when the experiment started, which fixed the memory budget), {env['os']}, "
            f"Python {env['python']}, NumPy {env['numpy']} ({env['blas']}), Dask/Distributed {env['dask']}. BLAS threads pinned to {env['blas_threads_env']} "
            "(OPENBLAS/OMP/MKL_NUM_THREADS) in the driver and inherited by workers; Dask threads per worker = 1. "
            "<b>Workers = processes = 1 per configuration slot; physical machines = 1.</b> Running multiple workers locally "
            "measures framework/process overhead and inter-process (loopback) transfer, not inter-machine network performance nor multi-machine memory capacity. "
            "Local execution was used because no cluster was available; the instructor's permission for local execution is not confirmed in the materials and should be verified before submission.")]
    s += [P("<b>3.4 Inputs and validation.</b> <font face='Courier'>numpy.random.default_rng(42)</font> (PCG64), A then B, uniform [0,1), float64, generated in Python once per n "
            "and passed to every method (same logical inputs). Validation before benchmarking (<font face='Courier'>validate.py</font>): known 2x2 product, identity, zero, "
            "pure-Python triple loop vs NumPy (n=40), threaded and distributed results vs reference for n=1, 2, 7, 257 (n&lt;workers, n not divisible by workers, odd n), identity and zero through the cluster &mdash; all passed. "
            "Every benchmark result was also compared with a reference product with rtol=atol=1e-10 (float64 eps&asymp;1.1e-16; the worst-case dot-product error for n&le;4000 is about n&middot;eps&asymp;4e-13, "
            "leaving two orders of magnitude of headroom); all measured runs passed; the maximum relative error was "
            f"{d.max_rel_err.max():.1e}.")]
    s += [P("<b>3.5 Timing protocol.</b> <i>End-to-end boundary (all methods):</i> A and B resident in driver memory &rarr; C resident in driver memory. Included: for Dask, scatter, task scheduling, compute, gather and copy into C; "
            "for baselines, output allocation and kernel. Excluded: input generation, reference computation, validation and <b>cluster start-up</b> (reported separately, median 0.9 s for 1 worker to 1.9 s for 16). "
            "Timer: <font face='Courier'>time.perf_counter</font> (monotonic). 1 warm-up run (discarded, not in CSV) and 5 measured repetitions per configuration; all 5 raw values are in the CSV; "
            "tables report median and IQR. Dask phases are measured sequentially, each ending with a blocking wait, so <b>scatter + compute + gather = end-to-end by construction and do not overlap</b>; "
            "'compute' is wall time from task submission until all futures finish and includes scheduler latency, while 'kernel' is the in-worker <font face='Courier'>perf_counter</font> time of the multiplication "
            "(the slowest worker, which determines the critical path). A fresh Python process (and cluster) is used for each (n, workers) pair so that memory peaks are not contaminated.")]
    s += [P("<b>3.6 Budgets and memory metric.</b> Maximum 300 s per configuration (never reached; longest was 2.4 s). Memory budget 1.7 GB total, because the machine had about 1.85 GB free. Planning estimate: "
            "4&middot;8n<super>2</super> (driver A, B, reference, C) + p&middot;8n<super>2</super> (B replicas) + 2&middot;8n<super>2</super> (A and C blocks) + p&middot;80 MB (idle worker interpreter). "
            "Configurations above budget were <b>not run</b> (no time is reported for them): " +
            "; ".join(f"n={int(r.n)}, w={int(r.workers)} (est. {r.est_memory_gb} GB)" for r in sk.itertuples()) +
            ". The largest size meeting the limits is n=4000 (4 workers); n=4000 with 8 workers was the first one attempted above budget. "
            "Memory metric: Windows <i>peak working set</i> (<font face='Courier'>psutil.Process().memory_info().peak_wset</font>), per worker process and for the driver (includes the inputs and the reference matrix). "
            "It is a resident-memory peak, not heap or array storage. Baseline peaks are cumulative over the process lifetime (n ascending in one process), so they are an upper bound for smaller n.")]

    s += [P("4. Results", H1)]
    rows = [["n", "workers", "e2e median (s)", "IQR", "scatter", "compute", "gather", "kernel max", "speedup e2e", "eff. e2e", "speedup kernel", "overhead %"]]
    for r in d.itertuples():
        rows.append([int(r.n), int(r.workers), f"{r.total_med:.3f}", f"{r.total_iqr:.3f}", f"{r.scatter_med:.3f}", f"{r.compute_med:.3f}",
                     f"{r.gather_med:.3f}", f"{r.task_max_med:.3f}", f"{r.speedup_e2e:.2f}", f"{r.efficiency_e2e:.2f}", f"{r.speedup_kernel:.2f}", f"{r.overhead_pct:.0f}"])
    s += [P("<b>Table 1.</b> Distributed (Dask) strong scaling. Times in seconds, median of 5 repetitions (1 warm-up discarded). Phases are sequential, so scatter+compute+gather&asymp;e2e (medians need not sum exactly). "
            "Speedup vs the sequential median of Table 2. 'overhead %' = (e2e &minus; compute)/e2e = share of transfer. 'kernel max' is in-worker multiplication time of the slowest worker.", SM),
          table(rows, fs=6.6)]
    rows = [["n", "sequential (s)", "IQR", "threads 2", "threads 4", "threads 8", "threads 16", "speedup@8 thr", "peak WS (MB)"]]
    for n in sorted(seq.index):
        p = ps[ps.n == n].set_index("workers")
        rows.append([n, f"{seq.loc[n, 'med']:.4f}", f"{seq.loc[n, 'iqr']:.4f}"] + [f"{bs[(bs.method=='parallel_threads')&(bs.n==n)&(bs.workers==k)].med.iloc[0]:.4f}" for k in (2, 4, 8, 16)] +
                    [f"{p.loc[8, 'speedup']:.2f}", f"{seq.loc[n, 'peak_mb']:.0f}"])
    s += [Spacer(1, 8), P("<b>Table 2.</b> Baselines. Kernel time (s), median of 5; same boundary as the end-to-end Dask time without transfers.", SM), table(rows, fs=7)]
    s += [Spacer(1, 6), img("fig1_scaling.png", 17 * cm),
          P("<b>Figure 1.</b> Left: strong-scaling speedup over the sequential baseline (solid: Dask end-to-end; dashed: shared-memory threads; dotted: ideal). Right: Dask end-to-end time; dotted lines are the sequential times. Log-log axes; configurations above the memory budget are absent.", C),
          img("fig2_phases.png", 17.3 * cm),
          P("<b>Figure 2.</b> Dask end-to-end time split into its three sequential phases (medians). Dashed: sequential baseline for the same n.", C)]
    mrow = [["n", "workers", "bytes moved (MB)", "scatter MB/s", "gather MB/s", "peak WS / worker (MB)", "driver peak WS (MB)"]]
    for r in d.itertuples():
        mrow.append([int(r.n), int(r.workers), f"{r.bytes_moved_mb:.0f}", f"{r.scatter_MBps:.0f}", f"{r.gather_MBps:.0f}", f"{r.worker_peak_mb:.0f}", f"{r.driver_peak_mb:.0f}"])
    s += [P("<b>Table 3.</b> Data volume, effective transfer rate and memory. Bytes moved = (p+2)&middot;8n<super>2</super> (computed from the algorithm, not measured on a network). Throughput = bytes/phase time, "
            "so it includes serialisation and scheduling; it is loopback inter-process throughput, not network bandwidth. Peak working set = max over workers.", SM), table(mrow, fs=7)]
    s += [Spacer(1, 6), img("fig3_memory.png", 9.5 * cm), P("<b>Figure 3.</b> Peak working set per worker process versus workers (max over workers).", C)]

    t = d.set_index(["n", "workers"])
    s += [P("5. Discussion", H1),
          P(f"<b>Computation (H1: partly supported).</b> The in-worker kernel scales well up to 4 workers (kernel efficiency {t.loc[(3000,4),'efficiency_kernel']:.2f} at n=3000; {t.loc[(4000,4),'efficiency_kernel']:.2f} at n=4000) "
            f"and degrades at 8 ({t.loc[(3000,8),'efficiency_kernel']:.2f} at n=3000). The shared-memory threaded baseline shows the same pattern with no transfers at all (speedup ~4 at 8 threads, "
            f"no gain at 16), so the ceiling is a property of this machine and kernel, not of Dask. Plausible causes (not isolated here, no counters were collected): memory bandwidth, "
            f"the 8 physical cores being shared with SMT siblings, and power/thermal limits of a laptop CPU. With one worker, the kernel equals sequential time (ratio {t.loc[(3000,1),'speedup_kernel']:.2f}), "
            "so the worker process itself adds no kernel penalty."),
          P(f"<b>End-to-end (H2: supported; H3: supported).</b> End-to-end speedup is far lower than kernel speedup. With one worker the distributed run is slower than sequential "
            f"({t.loc[(3000,1),'speedup_e2e']:.2f}x at n=3000) because B and A are copied to the worker and C back, 13&ndash;45% of the time. The share of transfer falls with n "
            f"({t.loc[(1000,1),'overhead_pct']:.0f}% at n=1000 vs {t.loc[(4000,1),'overhead_pct']:.0f}% at n=4000, one worker) as predicted by the O(n<super>3</super>)/O(n<super>2</super>) argument. "
            f"Scatter grows with the worker count because the B replica traffic is p&middot;8n<super>2</super> bytes (n=3000: {t.loc[(3000,1),'scatter_med']:.2f} s with 1 worker, {t.loc[(3000,8),'scatter_med']:.2f} s with 8), "
            f"while compute shrinks, so beyond 4 workers end-to-end time stops improving (n=3000: {t.loc[(3000,4),'total_med']:.2f} s at 4, {t.loc[(3000,8),'total_med']:.2f} s at 8). "
            f"For n=1000 the problem (35 ms sequentially) is too small: every Dask configuration is slower than sequential, and 16 workers (more than the 8 physical cores) cost {t.loc[(1000,16),'total_med']:.2f} s. "
            f"Dask beats sequential only for n&ge;2000 with 2&ndash;4 workers (best {best.speedup_e2e:.2f}x). It also never matched the shared-memory baseline, which is the cost of crossing process boundaries."),
          P("<b>Memory.</b> Per-worker peak working set falls with workers (n=3000: "
            f"{t.loc[(3000,1),'worker_peak_mb']:.0f} MB &rarr; {t.loc[(3000,8),'worker_peak_mb']:.0f} MB) but far less than 1/p, because each worker keeps the full replica of B (72 MB at n=3000) plus the interpreter and BLAS "
            "baseline; the sum over workers grows with p. Row-block with broadcast therefore does not remove the single-node memory requirement for B: it does not allow matrices whose B does not fit on one worker. "
            "Larger-than-one-machine inputs were not attempted (an extension that needs infrastructure)."),
          P("<b>Limitations.</b> (1) Single machine: transfer cost is IPC over loopback and cannot be extrapolated to a real network; network cost was not isolated and no network measurements were invented. "
            "(2) Free RAM (~1.85 GB of 16 GB) capped sizes at n=4000 and prevented 8/16 workers at the largest sizes; scaling for n&ge;4000 with 8 workers is untested. "
            "(3) Only five repetitions, one machine-session, other processes running; IQRs are small (&le;0.08 s) except for oversubscribed configurations (n=1000, 16 workers: IQR 0.12 s on a 0.21 s median). "
            "(4) Weak scaling was not studied. (5) Dask internals (serialisation, scheduler latency) are not separated: 'compute' wall minus 'kernel max' is an upper bound of scheduling overhead. "
            "(6) Peak working set is Windows-specific and not comparable to heap or array size. (7) Phase 'scatter' includes B replication and A distribution together. "
            "(8) 2-D partitioning (which would reduce replicated traffic) was not implemented."),
          P("6. Conclusions", H1),
          P(f"Within the tested scope (one laptop, float64, n=1000&ndash;4000, up to 8 workers at n&le;3000 and 4 at n=4000): the multiplication kernel parallelises well across worker processes up to about 4&ndash;8 workers, "
            f"but a row-block Dask implementation with broadcast of B yields at most {best.speedup_e2e:.2f}x end-to-end over sequential BLAS because transfer and framework overhead grow with workers and are significant at these sizes. "
            "Whether the same trade-off holds with a real network, larger n, or a 2-D decomposition remains unresolved and requires a multi-machine experiment."),
          P("7. References and materials", H1),
          P("[1] A. Grama, A. Gupta, G. Karypis and V. Kumar, <i>Introduction to Parallel Computing</i>, 2nd ed., Addison-Wesley, 2003 (matrix multiplication decompositions; speedup and efficiency).<br/>"
            "[2] R. A. van de Geijn and J. Watts, &ldquo;SUMMA: Scalable universal matrix multiplication algorithm,&rdquo; <i>Concurrency: Practice and Experience</i>, vol. 9, no. 4, pp. 255&ndash;274, 1997.<br/>"
            "[3] Dask Development Team, &ldquo;Dask Distributed documentation,&rdquo; https://distributed.dask.org (accessed Nov. 2026).<br/>"
            "[4] C. R. Harris et al., &ldquo;Array programming with NumPy,&rdquo; <i>Nature</i>, vol. 585, pp. 357&ndash;362, 2020.", SM),
          P("<b>Materials</b> (folder <font face='Courier'>assignment4/</font>, to be published in the course GitHub repository): <font face='Courier'>src/</font> (code), <font face='Courier'>results/*_raw.csv</font> (all raw measurements), "
            "<font face='Courier'>results/summary_*.csv</font>, <font face='Courier'>results/environment.json</font>, <font face='Courier'>results/skipped.csv</font>, <font face='Courier'>figures/</font>, <font face='Courier'>README.md</font> (exact commands), <font face='Courier'>requirements.txt</font>. "
            "Reproduce: <font face='Courier'>python src/run_all.py &amp;&amp; python src/analyze.py &amp;&amp; python src/make_report.py</font>.", SM),
          P("<b>Generative-AI disclosure.</b> Claude (Anthropic; Sonnet 5.5, with Haiku used to read the assignment PDF) was used to plan the experiment, write the code and draft this report. "
            "All numbers in tables and figures are generated by the included scripts from the raw CSVs; the author is responsible for understanding the code, verifying the measurements and references, and defending the conclusions.", SM)]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Assignment 4 - Distributed Matrix Multiplication", author="Javier Bolivar Garcia-Izquierdo")
    doc.build(s)
    print("report ->", OUT)


if __name__ == "__main__":
    main()
