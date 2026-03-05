import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics


@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real * z_real + z_imag * z_imag
        if z_sq > 4.0:
            return i

        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real * z_real - z_imag * z_imag + c_real

    return max_iter


@njit
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


if __name__ == "__main__":
    N = 1000
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 100

    # Serial execution
    start_time = time.time()
    mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    serial_time = time.time() - start_time
    print(f"Serial execution time: {serial_time:.4f} seconds")
