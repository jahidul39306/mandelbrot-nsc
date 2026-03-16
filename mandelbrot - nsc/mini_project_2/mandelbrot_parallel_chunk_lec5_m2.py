import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics
import sys, os
sys.path.append(os.path.dirname(__file__))  # so workers can find this module

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real * z_real + z_imag * z_imag
        if z_sq > 4.0:
            return i

        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real * z_real - z_imag * z_imag + c_real

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


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None
):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    max_iter = 100
    best_workers = 12
    t_baseline = None

    print(f"{'n_chunks':<12} {'time (s)':<12} {'vs. 1x':<12} {'LIF':<10}")
    print("-" * 46)

    for multiplier in [1, 2, 4, 8, 16, 32]:
        n_chunks = multiplier * best_workers

        # warmup for this configuration
        mandelbrot_parallel(
            N,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iter,
            n_workers=best_workers,
            n_chunks=n_chunks,
        )

        times = []

        for _ in range(7):
            t0 = time.perf_counter()
            mandelbrot_parallel(
                N,
                x_min,
                x_max,
                y_min,
                y_max,
                max_iter,
                n_workers=best_workers,
                n_chunks=n_chunks,
            )
            times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        if t_baseline is None:
            t_baseline = t_par
        vs_1x = t_baseline / t_par
        lif = (best_workers * t_par / t_baseline) - 1
        label = f"{multiplier}× n_workers"

        print(f"{label:<16} {t_par:<12.4f} {vs_1x:<12.4f} {lif:<10.4f}")
