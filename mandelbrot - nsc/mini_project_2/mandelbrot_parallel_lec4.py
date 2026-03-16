import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics


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


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
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

    # Serial execution
    start_time = time.perf_counter()
    mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    serial_time = time.perf_counter() - start_time
    print(f"Serial execution time: {serial_time:.4f} seconds")

    # Parallel execution
    start_time = time.perf_counter()
    mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers=4)
    parallel_time = time.perf_counter() - start_time
    print(f"Parallel execution time: {parallel_time:.4f} seconds")

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    for n_workers in range(1, os.cpu_count() + 1):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            mandelbrot_parallel(
                N,
                x_min,
                x_max,
                y_min,
                y_max,
                max_iter,
                n_workers=n_workers
            )
            times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        speedup = t_serial / t_par
        print(
            f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%"
        )
