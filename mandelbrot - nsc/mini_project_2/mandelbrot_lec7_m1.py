from numba import njit
from dask import delayed
from dask.distributed import Client
import dask, numpy as np, time, statistics


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


def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
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


if __name__ == "__main__":
    N, max_iter = 4096, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    client = Client("tcp://10.92.0.31:8786")
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    # Correctness check once
    ref = mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    assert np.array_equal(ref, result), "Dask result differs from serial!"
    print("Correctness check passed ✅\n")

    # --- Chunk sweep ---
    chunk_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    n_repeats = 3

    results = []
    for n_chunks in chunk_counts:
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)
        median_time = statistics.median(times)
        results.append((n_chunks, median_time))
        print(f"n_chunks={n_chunks:4d}  time={median_time:.4f}s")

    # Summary table
    baseline_time = results[0][1]  # time at n_chunks=1 as baseline
    print("\nn_chunks | Time (s) | vs 1x  | Speedup")
    print("-" * 45)
    for n_chunks, t in results:
        ratio = t / baseline_time
        speedup = baseline_time / t
        print(f"{n_chunks:8d} | {t:.4f}   | {ratio:.3f}  | {speedup:.3f}x")

    best_chunks, best_time = min(results, key=lambda x: x[1])
    print(f"\nBest: n_chunks={best_chunks} → {best_time:.4f}s")

    client.close()