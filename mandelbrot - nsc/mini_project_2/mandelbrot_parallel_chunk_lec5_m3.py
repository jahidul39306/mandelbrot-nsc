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
    N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None, pool=None
):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    if pool is not None:
        parts = pool.map(_worker, chunks)
    else:
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)  # un-timed warm-up: Numba JIT in workers
            parts = pool.map(_worker, chunks)
    return np.vstack(parts)


def serial_fraction(speedup, p):
    if p == 1:
        return 0
    return ((1 / speedup) - (1 / p)) / (1 - (1 / p))


if __name__ == "__main__":
    N, max_iter = 1024, 100
    n_workers = 12  # adjust to your L04 optimum
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)  # warm up JIT
    # Serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")

    # Chunk-count sweep (M2): one Pool per config
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)  # warm-up: load JIT cache in workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(
                    N,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                    max_iter,
                    n_workers=n_workers,
                    n_chunks=n_chunks,
                    pool=pool,
                )
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        lif = n_workers * t_par / t_serial - 1
        print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")

    print("\nWorkers  Time(s)  Speedup  Efficiency  s")

    for n_workers in range(1, 13):

        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)  # warm-up

            times = []
            for _ in range(3):
                t0 = time.perf_counter()

                mandelbrot_parallel(
                    N,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                    max_iter,
                    n_workers=n_workers,
                    n_chunks=8 * n_workers,
                    pool=pool,
                )

                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)

        speedup = t_serial / t_par
        efficiency = speedup / n_workers
        s = serial_fraction(speedup, n_workers)

        print(f"{n_workers:3d} {t_par:8.3f} {speedup:8.2f} {efficiency:10.2f} {s:6.3f}")
