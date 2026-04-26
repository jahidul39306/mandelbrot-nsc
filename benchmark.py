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
    fn(*args)  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    n_workers = 8
    resolutions = [1024, 4096]
    results = {}

    for res in resolutions:
        print(f"\n--- Benchmarking {res}x{res} ---")
        width, height = res, res

        args = (-2, 1, -1.5, 1.5, width, height)
        args_parallel = (width, -2, 1, -1.5, 1.5, 100, n_workers, n_workers * 2)
        args_dask_local = (width, -2, 1, -1.5, 1.5, 100, 16) # 16 is the best n_chunks for dask local with 8 workers
        args_dask_strato = (width, -2, 1, -1.5, 1.5, 100, 32) # 32 is the best n_chunks for dask strato

        t_naive = bench(mandelbrot_naive, *args)
        t_numpy = bench(mandelbrot_numpy, *args)
        t_hybrid = bench(mandelbrot_hybrid, *args)
        t_naive_numba = bench(mandelbrot_naive_numba, *args)
        t_parallel = bench(mandelbrot_parallel, *args_parallel)
        t_dask_local = bench(mandelbrot_dask, *args_dask_local)
        t_dask_strato = bench(mandelbrot_dask_strato, *args_dask_strato)

        results[res] = {
            "Naive": t_naive,
            "NumPy": t_numpy,
            "Hybrid": t_hybrid,
            "Numba": t_naive_numba,
            "Numba + Parallel": t_parallel,
            "Dask local": t_dask_local,
            "Dask strato": t_dask_strato,
        }

    # Build output string
    methods = list(next(iter(results.values())).keys())
    col_w = 18

    lines = []
    lines.append("===== BENCHMARK RESULTS (median seconds) =====")
    header = f"{'Method':<20}" + "".join(f"{f'{r}x{r}':>{col_w}}" for r in resolutions)
    lines.append(header)
    lines.append("-" * len(header))
    for method in methods:
        row = f"{method:<20}" + "".join(
            f"{results[r][method]:>{col_w}.3f}" for r in resolutions
        )
        lines.append(row)

    lines.append("\n===== SPEEDUP (relative to Naive) =====")
    lines.append(header)
    lines.append("-" * len(header))
    for method in methods:
        row = f"{method:<20}" + "".join(
            f"{results[r]['Naive'] / results[r][method]:>{col_w}.2f}x"
            for r in resolutions
        )
        lines.append(row)

    output = "\n".join(lines)

    # Print to console
    print("\n\n" + output)

    # Save to file
    with open("benchmark_results.txt", "w") as f:
        f.write(output)

    print("\nResults saved to benchmark_results.txt")
