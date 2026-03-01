"""
Mandelbrot Set Generator
Author : Md Jahidul Islam Noor
Course : Numerical Scientific Computing 2026
"""

from os import name
import statistics
from numba import njit

import numpy as np
import time


@njit
def mandelbrot_point_numba(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z**2 + c
    return max_iter


def mandelbrot_hybrid(
    xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5, width=1024, height=1024, max_iter=100
):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=int)

    for i in range(height):
        for k in range(width):
            c = x_vals[k] + 1j * y_vals[i]
            result[i, k] = mandelbrot_point_numba(c, max_iter)

    return result


@njit
def mandelbrot_naive_numba(
    xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5, width=1024, height=1024, max_iter=100
):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for k in range(width):
            c = x_vals[k] + 1j * y_vals[i]
            z = 0j
            n = 0
            while n < max_iter and (z.real * z.real + z.imag * z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, k] = n
    return result


def bench(fn, *args, runs=5):
    fn(*args)  # extra warm -up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    # Warm up the JIT compiler
    _ = mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64)
    _ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)

    t_hybrid = bench(mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024)
    t_full = bench(mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024)

    print(f" Hybrid : { t_hybrid :.3f}s")
    print(f" Fully compiled : { t_full :.3f}s")
    print(f" Ratio : { t_hybrid / t_full :.1f}x")
