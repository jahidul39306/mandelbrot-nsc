from numba import njit
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask
import numpy as np
import time, statistics
import matplotlib.pyplot as plt


@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
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
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)

    return out


def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter, n_chunks):
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


def measure_time(fn, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":

    N, max_iter = 4096, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # Start cluster ONCE
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    # Warm-up Numba in ALL workers
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    # ---------------------------
    # Sweep chunk counts
    # ---------------------------
    T1 = None
    chunk_values = [1, 2, 4, 8, 16, 32, 64, 128]

    results = []

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
        p = 8  # number of workers

        # LIF formula
        LIF = p * (Tp / T1) - 1

        results.append((n_chunks, Tp, vs1x, speedup, LIF))

        print(f"{n_chunks:8d} | {Tp:8.4f} | {vs1x:6.3f} | {speedup:7.3f} | {LIF:6.3f}")

    # ---------------------------
    # Find optimal
    # ---------------------------
    best = min(results, key=lambda x: x[1])
    print("\nOptimal:")
    print(f"n_chunks = {best[0]}, time = {best[1]:.4f}, LIF = {best[4]:.4f}")

    # ---------------------------
    # Plot
    # ---------------------------
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
