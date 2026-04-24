from numba import njit
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask
import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple


@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """
    Compute the number of iterations for a single point in the Mandelbrot set.

    Parameters
    ----------
    c_real : float
        Real part of the complex number.
    c_imag : float
        Imaginary part of the complex number.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Number of iterations before divergence, or max_iter if bounded.
    """
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real * z_real + z_imag * z_imag
        if z_sq > 4.0:
            return i
        z_real, z_imag = (
            z_real * z_real - z_imag * z_imag + c_real,
            2.0 * z_real * z_imag + c_imag,
        )
    return max_iter


@njit(cache=True)
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
) -> np.ndarray:
    """
    Compute a chunk of rows of the Mandelbrot set.

    Parameters
    ----------
    row_start : int
        Starting row index (inclusive).
    row_end : int
        Ending row index (exclusive).
    N : int
        Resolution (width and height).
    x_min : float
        Minimum x-coordinate.
    x_max : float
        Maximum x-coordinate.
    y_min : float
        Minimum y-coordinate.
    y_max : float
        Maximum y-coordinate.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        2D array of iteration counts for the chunk.
    """
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)

    return out


def mandelbrot_dask(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
    n_chunks: int,
) -> np.ndarray:
    """
    Compute the Mandelbrot set using Dask parallelism.

    Parameters
    ----------
    N : int
        Resolution (width and height).
    x_min : float
        Minimum x-coordinate.
    x_max : float
        Maximum x-coordinate.
    y_min : float
        Minimum y-coordinate.
    y_max : float
        Maximum y-coordinate.
    max_iter : int
        Maximum number of iterations.
    n_chunks : int
        Number of chunks to divide the computation into.

    Returns
    -------
    np.ndarray
        Full Mandelbrot set as a 2D array.
    """
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0

    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end

    parts = dask.compute(*tasks)
    return np.vstack(parts)


def measure_time(fn: Callable[[], None], runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_experiment() -> None:
    """
    Run chunk-scaling experiment and plot results.
    """
    N, max_iter = 4096, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    # Warm-up
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    T1: float | None = None
    chunk_values = [1, 2, 4, 8, 16, 32, 64, 128]

    results: List[Tuple[int, float, float, float, float]] = []

    print("n_chunks | time (s) | vs 1x | speedup | LIF")
    print("-" * 50)

    for n_chunks in chunk_values:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks)
            times.append(time.perf_counter() - t0)

        Tp = statistics.median(times)

        if T1 is None:
            T1 = Tp
            print(f"Baseline (1 chunk): {T1:.4f} seconds")

        speedup = T1 / Tp
        vs1x = Tp / T1
        p = 8

        LIF = p * (Tp / T1) - 1

        results.append((n_chunks, Tp, vs1x, speedup, LIF))

        print(f"{n_chunks:8d} | {Tp:8.4f} | {vs1x:6.3f} | {speedup:7.3f} | {LIF:6.3f}")

    best = min(results, key=lambda x: x[1])
    print("\nOptimal:")
    print(f"n_chunks = {best[0]}, time = {best[1]:.4f}, LIF = {best[4]:.4f}")

    x = [r[0] for r in results]
    y = [r[1] for r in results]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xscale("log")
    plt.xlabel("n_chunks (log scale)")
    plt.ylabel("Time (s)")
    plt.title("Dask Chunk Sweep")
    plt.grid()

    plt.savefig("dask_chunk_sweep.png")
    plt.show()

    client.close()
    cluster.close()


if __name__ == "__main__":
    run_experiment()
