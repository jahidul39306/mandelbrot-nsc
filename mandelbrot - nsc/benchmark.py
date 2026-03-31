import time, statistics
from mini_project_1.mandelbrot_naive_numba import (
    mandelbrot_hybrid,
    mandelbrot_naive_numba,
)
from mini_project_1.mandelbrot_naive import mandelbrot_naive
from mini_project_1.mandelbrot_numpy import mandelbrot_numpy
from mini_project_2.mandelbrot_parallel_chunk_lec5_m2 import mandelbrot_parallel
from mini_project_2.mandelbrot_parallel_chunk_lec6_m1 import mandelbrot_dask
from mini_project_2.mandelbrot_lec7_m1 import mandelbrot_dask as mandelbrot_dask_strato


def bench(fn, *args, runs=5):
    fn(*args)  # warm -up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    width, height = 4096, 4096
    n_workers = 8
    args = (-2, 1, -1.5, 1.5, width, height)
    args_parallel = (
        width,
        -2,
        1,
        -1.5,
        1.5,
        100,
        n_workers,
        n_workers * 2,
    )  # best result got from n_chunks = 2 * n_workers
    args_dask_local = (
        width,
        -2,
        1,
        -1.5,
        1.5,
        100,
        4,
    )  # 4 is the best n_chunks for dask local with 8 workers
    args_dask_strato = (
        width,
        -2,
        1,
        -1.5,
        1.5,
        100,
        4,
    )

    t_naive = bench(mandelbrot_naive, *args)
    t_numpy = bench(mandelbrot_numpy, *args)
    t_hybrid = bench(mandelbrot_hybrid, *args)
    t_naive_numba = bench(mandelbrot_naive_numba, *args)
    t_mandelbrot_parallel = bench(mandelbrot_parallel, *args_parallel)
    t_mandelbrot_dask_local = bench(mandelbrot_dask, *args_dask_local)
    t_mandelbrot_dask_strato = bench(mandelbrot_dask_strato, *args_dask_strato)

    print(f"Naive: {t_naive:.3f} seconds")
    print(f"NumPy: {t_numpy:.3f} seconds, speedup: {t_naive / t_numpy :.2f}x")
    print(f"Hybrid: {t_hybrid:.3f} seconds, speedup: {t_naive / t_hybrid :.2f}x")
    print(
        f"Numba: {t_naive_numba:.3f} seconds, speedup: {t_naive / t_naive_numba :.2f}x"
    )
    print(
        f"Numba + Parallel: {t_mandelbrot_parallel:.3f} seconds, speedup: {t_naive / t_mandelbrot_parallel :.2f}x"
    )
    print(
        f"Dask local: {t_mandelbrot_dask_local:.3f} seconds, speedup: {t_naive / t_mandelbrot_dask_local :.2f}x"
    )
    print(
        f"Dask strato: {t_mandelbrot_dask_strato:.3f} seconds, speedup: {t_naive / t_mandelbrot_dask_strato :.2f}x"
    )
