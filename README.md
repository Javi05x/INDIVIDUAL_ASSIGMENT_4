# Assignment 4 - Distributed Matrix Multiplication

Author: Javier Bolivar Garcia-Izquierdo. BIG DATA 2026-2027.

Dask Distributed row-block matrix multiplication, strong scaling vs a sequential and a shared-memory parallel baseline.
**Environment label: local execution (1 machine, workers = processes).** It does not measure inter-machine network or multi-machine memory.

## Requirements
Windows 11, Python 3.12.3 (tested). Memory metric `peak_wset` is Windows-only.

```
python -m pip install -r requirements.txt
```

## Run (from the `assignment4/` folder)
```
python src/validate.py        # correctness tests (must print ALL VALIDATION TESTS PASSED)
python src/run_all.py         # validation + environment + baselines + all (n, workers) runs  (~3 min)
python src/analyze.py         # summary tables + figures
python src/make_report.py     # PDF in report/
```
Single configuration: `python src/run_benchmark.py distributed --n 2000 --w 4`.
Delete `results/*.csv` before a re-run (results are appended).

## Layout
- `src/common.py` input generator (`default_rng(42)`, float64, U[0,1)), tolerances (rtol=atol=1e-10), BLAS pinned to 1 thread.
- `src/distributed_mm.py` Dask implementation; `src/baselines.py` sequential and threaded baselines.
- `src/run_benchmark.py`, `src/run_all.py` protocol (1 warm-up + 5 reps, fresh process per configuration).
- `results/` raw CSVs, summaries, `environment.json`, `skipped.csv` (configurations above the declared memory budget; not run).
- `figures/`, `report/` outputs.

Inputs are regenerated deterministically by `make_inputs(n)`; no data files.
Assignments 1-3 materials were not reused (baselines were written here).
