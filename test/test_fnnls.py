import os

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import nnls
from scipy.linalg import lstsq
import pytest

from fnnlsEigen import fnnls, fnnlsf, CachePreComputeNNLS, CachePreComputeNNLSf


def plot_solution(Z, x, c, t, label):
    print()
    print("Solver:", label)
    print("time: {:.0f} ms.".format(1000 * t))
    print(f"Sum(c) = {np.sum(c)}")
    print(f"Residual norm: {np.linalg.norm(Z @ c - x)}")

    plt.figure(1)
    plt.plot(c, "-o", label=label)
    plt.xlabel("simulation mixture indices")
    plt.ylabel("volume concentrations")
    plt.figure(2)
    plt.plot(Z @ c - x, "-o", label=label)
    plt.xlabel("observation indices")
    plt.ylabel("residual")


@pytest.mark.parametrize("solver, dtype", [(fnnls, np.float64), (fnnlsf, np.float32)])
def test_eigen_fnnls(solver, dtype, visualise=False):

    DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(DIRECTORY, "test_data")

    Z = np.loadtxt(os.path.join(path, "Z.dat")).astype(dtype)
    x = np.loadtxt(os.path.join(path, "x.dat")).astype(dtype)

    print(f"Z: shape {Z.shape}, and dtype {Z.dtype}.")
    print(f"x: shape {x.shape}, and dtype {x.dtype}.")
    print(
        "About {:.1f} % of all elements are non-zero in Z.".format(
            100 * np.sum(np.abs(Z) > 0) / Z.size
        )
    )

    t = time.monotonic()
    c_scipy = nnls(Z, x, maxiter=10 * Z.shape[1])[0]
    t = time.monotonic() - t

    res_scipy_nnls = np.linalg.norm(Z @ c_scipy - x)

    if visualise:
        plot_solution(Z, x, c_scipy, t, label="scipy nnls")

    t = time.monotonic()
    c_eigen = solver(Z, x)
    t = time.monotonic() - t

    res_eigen_fnnls = np.linalg.norm(Z @ c_eigen - x)

    if visualise:
        plot_solution(Z, x, c_eigen, t, label=" fnnls using Eigen in C++ (with Eigen::ldlt)")
        plt.figure(1)
        plt.legend()
        plt.figure(2)
        plt.legend()
        plt.show()

    if dtype is np.float64:
        np.testing.assert_almost_equal(res_scipy_nnls, res_eigen_fnnls)
        np.testing.assert_array_almost_equal(c_scipy, c_eigen)
    else:
        np.testing.assert_almost_equal(res_scipy_nnls, res_eigen_fnnls, decimal=6)
        np.testing.assert_array_almost_equal(c_scipy, c_eigen, decimal=2)


@pytest.mark.parametrize(
    "floating_class, solver, dtype",
    [(CachePreComputeNNLS, fnnls, np.float64), (CachePreComputeNNLSf, fnnlsf, np.float32)],
)
def test_precompute_same_input(floating_class, solver, dtype, laps=2):
    from time import monotonic

    DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(DIRECTORY, "test_data")

    Z = np.loadtxt(os.path.join(path, "Z.dat")).astype(dtype)
    x = np.loadtxt(os.path.join(path, "x.dat")).astype(dtype)

    t_start = monotonic()
    for _ in range(0, laps):
        direct_solve = solver(Z, x)
    t_direct = monotonic() - t_start

    t_start = monotonic()
    pc = floating_class()
    t_floating_class = monotonic() - t_start

    t_start = monotonic()
    for _ in range(0, laps):
        precompute_solve = pc.fnnls(Z, x)
    t_precompute = monotonic() - t_start

    np.testing.assert_almost_equal(direct_solve, precompute_solve)
    print(
        f"Direct solver took {t_direct} and precomputed"
        f" solver took {t_floating_class + t_precompute}"
        f" for {laps} number of repetitions."
    )
    assert t_direct > t_floating_class + t_precompute


@pytest.mark.parametrize(
    "floating_class, dtype, rtol",
    [(CachePreComputeNNLS, np.float64, 1e-14), (CachePreComputeNNLSf, np.float32, 1e-7)],
)
def test_precompute_varying_input(floating_class, dtype, rtol):
    """This test attempts to probe whether the cached ZTZ matrix is recomputed
    under various conditions. However, since ZTZ itself cannot be directly
    accessed, the asserts are performed indirectly through changes in the
    result."""

    Z = np.array([[4.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=dtype)
    x = np.array([5.0, 2.0, 1.0], dtype=dtype)

    pc = floating_class()
    first_solution = pc.fnnls(Z, x)

    # Some cases where ZTZ should be recomputed for both dtypes
    for i in range(5, 0, -1):
        delta = 0.1 ** (i)
        Z_new = Z.copy()
        Z_new[0, 2] -= delta
        precompute_solve = pc.fnnls(Z_new, x)
        assert not np.allclose(first_solution, precompute_solve, rtol)

    # Some cases where ZTZ should be recomputed only for np.float64
    print(dtype)
    for i in range(8, 6, -1):
        delta = 0.1 ** (i)
        Z_new = Z.copy()
        Z_new[0, 2] -= delta
        precompute_solve = pc.fnnls(Z_new, x)
        if dtype == np.float32:
            assert np.allclose(first_solution, precompute_solve, rtol)
        else:
            assert not np.allclose(first_solution, precompute_solve, rtol)

    # Some cases where ZTZ should not be recomputed for neither dtype
    for i in range(12, 8, -1):
        delta = 0.1 ** (i)
        Z_new = Z.copy()
        Z_new[0, 2] -= delta
        precompute_solve = pc.fnnls(Z_new, x)
        assert np.allclose(first_solution, precompute_solve, rtol)

    # Restore original Z, then change shape
    pc.fnnls(Z, x)
    with pytest.raises(RuntimeError):
        precompute_solve = pc.fnnls(Z, x[:-1])
    pc.fnnls(Z, x)
    with pytest.raises(RuntimeError):
        precompute_solve = pc.fnnls(Z[1::, :], x)
    pc.fnnls(Z, x)
    precompute_solve = pc.fnnls(Z[:, :-1].copy(), x)
    assert not np.array_equal(first_solution, precompute_solve)


@pytest.mark.parametrize("solver, dtype", [(fnnls, np.float64), (fnnlsf, np.float32)])
def test_least_square(solver, dtype):
    # A problem that is non-solvable, but has positive LS solution
    Z = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=dtype)
    x = np.array([1.0, 1.0, 0.0], dtype=dtype)

    d = solver(Z, x)

    assert np.allclose(d, [2.0 / 3.0, 2.0 / 3.0])


@pytest.mark.parametrize("solver, dtype", [(fnnls, np.float64), (fnnlsf, np.float32)])
def test_non_negative_least_square(solver, dtype):
    # A problem that is non-solvable, and has negative LS solution
    Z = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]], dtype=dtype)
    x = np.array([1.0, 1.0, 0.0], dtype=dtype)

    d = solver(Z, x)

    assert np.allclose(d, [2.0 / 3.0, 2.0 / 3.0, 0.0])


@pytest.mark.parametrize("solver, dtype", [(fnnls, np.float64), (fnnlsf, np.float32)])
def test_inner_loop(solver, dtype):
    # A problem that forces the "inner" loop to "re-impose" positive LS solution
    Z = np.array([[4.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=dtype)
    x = np.array([5.0, 2.0, 1.0], dtype=dtype)

    d = solver(Z, x)

    assert np.allclose(d, [0.0, 2.0, 1010.0 / 401.0])


@pytest.mark.parametrize("m", [1, 2, 3, 5, 50])
@pytest.mark.parametrize("n", [2, 3, 5, 100])
def test_underdetermined_system(m, n):
    """
    Test that get at the same solution as scipy's nnls for underdetermined system.
    However the solution is not the unique minium norm solution,
    but instead it seems to be one of the maximum norm solutions.

    """
    a = np.ones(m * n).reshape((m, n))
    b = np.ones(m)
    x_fnnls = fnnls(a, b)
    x_nnls = nnls(a, b)[0]
    x_lstsq = lstsq(a, b, cond=1e-14)[0]
    # The unique minimum norm solution
    x_min_norm = np.ones(n) / n
    # When n=3, there are three maximum norm solutions: [1,0,0], [0,1,0] and [0,0,1].
    x_non_unique_max_norm = np.zeros(n)
    x_non_unique_max_norm[0] = 1
    for x in [x_fnnls, x_nnls, x_lstsq, x_min_norm, x_non_unique_max_norm]:
        residuals = a @ x - b
        np.testing.assert_allclose(residuals, 0, atol=1e-12)
        print(x)
        print(np.linalg.norm(x))
        print()
    np.testing.assert_allclose(x_fnnls, x_non_unique_max_norm)
    np.testing.assert_allclose(x_nnls, x_non_unique_max_norm)
    np.testing.assert_allclose(x_lstsq, x_min_norm)
    assert np.sum(np.abs(x_non_unique_max_norm - x_min_norm)) > 0


@pytest.mark.parametrize("n", [2, 3, 100])
def test_underdetermined_system_negative_rhs(n):
    """
    Test that get at the same solution as scipy's nnls for underdetermined system.
    However the solution is not the unique minium norm solution,
    but instead it seems to be one of the maximum norm solutions.

    """
    a = np.ones(n).reshape((1, n))
    b = -1 * np.ones(1)
    x_fnnls = fnnls(a, b)
    x_nnls = nnls(a, b)[0]
    x_lstsq = lstsq(a, b, cond=1e-14)[0]
    # The unique minimum norm solution
    x_min_norm = -np.ones(n) / n
    x_ref = np.zeros(n)
    for x in [x_fnnls, x_nnls, x_ref]:
        residuals = a @ x - b
        np.testing.assert_allclose(residuals, 1, atol=1e-12)
        print(x)
        print(np.linalg.norm(x))
        print()
    for x in [x_lstsq, x_min_norm]:
        residuals = a @ x - b
        np.testing.assert_allclose(residuals, 0, atol=1e-12)
        print(x)
        print(np.linalg.norm(x))
        print()
    np.testing.assert_allclose(x_fnnls, x_ref)
    np.testing.assert_allclose(x_nnls, x_ref)
    np.testing.assert_allclose(x_lstsq, x_min_norm)
    assert np.sum(np.abs(x_ref - x_min_norm)) > 0


@pytest.mark.parametrize("solver, dtype", [(fnnls, np.float64), (fnnlsf, np.float32)])
def test_big_system(solver, dtype):
    m, n = 200, 150
    np.random.seed(1)
    a = np.random.randn(m, n).astype(dtype)
    b = np.random.randn(m).astype(dtype)
    x_fnnls = solver(a, b)
    x_nnls = nnls(a, b)[0]
    np.testing.assert_allclose(x_fnnls, x_nnls, atol=n * np.finfo(dtype).eps)


@pytest.mark.parametrize("solver, dtype", [(fnnls, np.float64), (fnnlsf, np.float32)])
def test_solvable_big_system(solver, dtype):
    m, n = 200, 150
    np.random.seed(1)
    a = np.random.randn(m, n).astype(dtype)
    x_initial = np.random.rand(n).astype(dtype)
    b = a @ x_initial
    x_fnnls = solver(a, b)
    x_nnls = nnls(a, b)[0]
    np.testing.assert_allclose(x_fnnls, x_nnls, atol=n * np.finfo(dtype).eps)
    np.testing.assert_allclose(a @ x_fnnls, b, atol=2 * n * np.finfo(dtype).eps)


def test_c_contiguous():
    # Test that sending in a Fortran major array yields a nice python exception.
    arr = np.arange(12).reshape(3, 4).astype(np.float64)
    with pytest.raises(ValueError):
        fnnls(arr.T, np.array([7, 8, 9], dtype=np.float64))


def test_incorrect_type():
    # Test that sending in a np.float32 yields a ValueError.
    arr = np.arange(12).reshape(3, 4).astype(np.float64)
    with pytest.raises(ValueError):
        fnnls(arr, np.array([7, 8, 9], dtype=np.float32))


def test_incorrect_dimensions():
    # Test that sending in the wrong dimensions yields a ValueError.
    arr = np.arange(12).reshape(3, 4).astype(np.float64)
    with pytest.raises(ValueError):
        fnnls(arr, np.array([[7, 8, 9], [34, 56, 12]], dtype=np.float64))


def test_contiguous():
    # Test that sending in a non-contiguous numpy array yields a nice python exception.
    arr = np.arange(12).reshape(3, 4).astype(np.float64)
    with pytest.raises(ValueError):
        fnnls(arr[:, 1:2], np.array([7, 8, 9], dtype=np.float64))
