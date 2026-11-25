"""Run the full experiment: validation, environment capture, baselines, then one fresh process per (n, w)."""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [1000, 2000, 3000, 4000]
WORKERS = [1, 2, 4, 8, 16]


def run(*args):
    subprocess.run([sys.executable, os.path.join(HERE, *args[:1]), *args[1:]], check=True, cwd=HERE)


if __name__ == "__main__":
    run("validate.py")
    run("run_benchmark.py", "env")
    run("run_benchmark.py", "baselines", "--n", *map(str, SIZES), "--k", *map(str, WORKERS[1:]))
    for n in SIZES:
        for w in WORKERS:
            run("run_benchmark.py", "distributed", "--n", str(n), "--w", str(w))
