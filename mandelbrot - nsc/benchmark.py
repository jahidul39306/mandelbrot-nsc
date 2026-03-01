import time, statistics
from mandelbrot_naive_numba import mandelbrot_hybrid, mandelbrot_naive_numba
from mandelbrot_naive import mandelbrot_naive
from mandelbrot_numpy import mandelbrot_numpy


def bench(fn, *args, runs=5):
    fn(*args)  # warm -up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    width, height = 1024, 1024
    args = (-2, 1, -1.5, 1.5, width, height)

    t_naive = bench(mandelbrot_naive, *args)
    t_numpy = bench(mandelbrot_numpy, *args)
    t_hybrid = bench(mandelbrot_hybrid, *args)
    t_naive_numba = bench(mandelbrot_naive_numba, *args)

    print(f"Naive: {t_naive:.3f} seconds")
    print(f"NumPy: {t_numpy:.3f} seconds, speedup: {t_naive / t_numpy :.2f}x")
    print(f"Hybrid: {t_hybrid:.3f} seconds, speedup: {t_naive / t_hybrid :.2f}x")
    print(
        f"Numba: {t_naive_numba:.3f} seconds, speedup: {t_naive / t_naive_numba :.2f}x"
    )
