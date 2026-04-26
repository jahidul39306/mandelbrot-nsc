import statistics
import time

import dask
import numpy as np
import numpy.typing as npt
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit


@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """Compute the Mandelbrot escape iteration count for a single pixel.

    Iterates the Mandelbrot recurrence using real-valued arithmetic and
    returns the step at which the orbit escapes the radius-2 boundary,
    or ``max_iter`` if it does not escape.

    Parameters
    ----------
    c_real : float
        Real component of the complex coordinate.
    c_imag : float
        Imaginary component of the complex coordinate.
    max_iter : int
        Maximum number of iterations before declaring the point inside
        the Mandelbrot set.

    Returns
    -------
    int
        Number of iterations before escape, or ``max_iter`` if the point
        did not escape.
    """
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
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
) -> npt.NDArray[np.int32]:
    """Compute a horizontal strip of the Mandelbrot grid (JIT-compiled).

    Evaluates :func:`mandelbrot_pixel` for every pixel in the row range
    [``row_start``, ``row_end``) across all ``N`` columns. Designed to be
    dispatched as an independent Dask task.

    Parameters
    ----------
    row_start : int
        Index of the first row to compute (inclusive).
    row_end : int
        Index of the last row to compute (exclusive).
    N : int
        Total grid size (number of columns and total rows).
    x_min : float
        Left boundary of the real axis.
    x_max : float
        Right boundary of the real axis.
    y_min : float
        Bottom boundary of the imaginary axis.
    y_max : float
        Top boundary of the imaginary axis.
    max_iter : int
        Maximum number of Mandelbrot iterations per pixel.

    Returns
    -------
    npt.NDArray[np.int32]
        2-D array of shape ``(row_end - row_start, N)`` containing the
        escape iteration count for each pixel in the strip.
    """
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out


def mandelbrot_dask(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    n_chunks: int = 32,
) -> npt.NDArray[np.int32]:
    """Compute the Mandelbrot set by distributing horizontal strips across Dask workers.

    Splits the ``N × N`` grid into ``n_chunks`` horizontal strips, wraps each
    in a :func:`dask.delayed` call to :func:`mandelbrot_chunk`, and computes
    all tasks in parallel via the active Dask scheduler. The resulting strips
    are stacked into a single array.

    Parameters
    ----------
    N : int
        Grid size; the output will be of shape ``(N, N)``.
    x_min : float
        Left boundary of the real axis.
    x_max : float
        Right boundary of the real axis.
    y_min : float
        Bottom boundary of the imaginary axis.
    y_max : float
        Top boundary of the imaginary axis.
    max_iter : int, optional
        Maximum number of Mandelbrot iterations per pixel. Default is 100.
    n_chunks : int, optional
        Number of horizontal strips to divide the grid into. Default is 32.

    Returns
    -------
    npt.NDArray[np.int32]
        2-D array of shape ``(N, N)`` containing the escape iteration
        count for each pixel.
    """
    chunk_size: int = max(1, N // n_chunks)
    tasks: list = []
    row: int = 0
    while row < N:
        row_end: int = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    # Warm-up: trigger JIT compilation in workers
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    ref: npt.NDArray[np.int32] = mandelbrot_chunk(
        0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter
    )
    result: npt.NDArray[np.int32] = mandelbrot_dask(
        N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter
    )
    assert np.array_equal(ref, result), "Dask result differs from serial!"
    print("Correctness check passed ✅")

    times: list[float] = []
    for _ in range(3):
        t0: float = time.perf_counter()
        result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    print(f"Dask local (n_chunks=32): {statistics.median(times):.3f} seconds")

    client.close()
    cluster.close()
