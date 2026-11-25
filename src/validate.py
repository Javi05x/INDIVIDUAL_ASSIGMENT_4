"""Correctness tests run BEFORE benchmarking (Assignment 4)."""
import common  # noqa: F401  (pins BLAS threads; must precede numpy)
import numpy as np
from common import make_inputs, naive_matmul, close


def main():
    from distributed_mm import DistributedMM
    from baselines import parallel_threads
    # 1. known answer, identity, zero
    A = np.array([[1., 2.], [3., 4.]]); B = np.array([[5., 6.], [7., 8.]])
    assert np.array_equal(A @ B, np.array([[19., 22.], [43., 50.]]))
    X = make_inputs(8)[0]
    assert np.array_equal(X @ np.eye(8), X) and np.array_equal(np.eye(8) @ X, X) and not (X @ np.zeros((8, 8))).any()
    # 2. pure-Python triple loop oracle vs numpy
    A, B = make_inputs(40)
    assert close(np.array(naive_matmul(A.tolist(), B.tolist())), A @ B)
    # 3. shared-memory parallel vs reference
    A, B = make_inputs(257)
    assert close(parallel_threads(A, B, 3), A @ B)
    # 4. distributed: odd n, n not divisible by w, n < w, identity, zero
    with DistributedMM(3) as dm:
        for n in (1, 2, 7, 257):
            A, B = make_inputs(n)
            C, _ = dm.multiply(A, B)
            assert close(C, A @ B), f"distributed mismatch n={n}"
        X = make_inputs(64)[0]
        C, _ = dm.multiply(X, np.eye(64)); assert np.array_equal(C, X)
        C, _ = dm.multiply(X, np.zeros((64, 64))); assert not C.any()
    print("ALL VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
