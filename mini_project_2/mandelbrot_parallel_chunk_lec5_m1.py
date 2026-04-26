import os
import statistics
import time
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
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
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real * z_real - z_imag * z_imag + c_real
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
    [``row_start``, ``row_end``) across all ``N`` columns. Intended to be
    called by worker processes as part of a parallel decomposition.

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


def mandelbrot_serial(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
) -> npt.NDArray[np.int32]:
    """Compute the full Mandelbrot grid serially using a single JIT-compiled chunk.

    Delegates the entire ``N × N`` grid to :func:`mandelbrot_chunk` in one
    call, with no parallelism. Useful as a baseline for benchmarking against
    :func:`mandelbrot_parallel`.

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

    Returns
    -------
    npt.NDArray[np.int32]
        2-D array of shape ``(N, N)`` containing the escape iteration
        count for each pixel.
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args: tuple) -> npt.NDArray[np.int32]:
    """Unpack arguments and delegate to :func:`mandelbrot_chunk`.

    This thin wrapper exists so that :func:`mandelbrot_chunk` can be called
    via :func:`multiprocessing.Pool.map`, which requires a single-argument
    callable.

    Parameters
    ----------
    args : tuple
        Positional arguments forwarded directly to :func:`mandelbrot_chunk`:
        ``(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter)``.

    Returns
    -------
    npt.NDArray[np.int32]
        2-D array of shape ``(row_end - row_start, N)`` as returned by
        :func:`mandelbrot_chunk`.
    """
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    n_workers: int = 4,
    n_chunks: int | None = None,
) -> npt.NDArray[np.int32]:
    """Compute the Mandelbrot grid in parallel using multiprocessing and Numba chunks.

    Divides the ``N × N`` grid into horizontal strips and distributes them
    across a :class:`multiprocessing.Pool`. Each worker calls the
    JIT-compiled :func:`mandelbrot_chunk`. A warm-up ``pool.map`` is issued
    first to trigger Numba JIT compilation in each worker process before
    the timed run.

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
    n_workers : int, optional
        Number of worker processes in the pool. Default is 4.
    n_chunks : int or None, optional
        Number of horizontal strips to divide the grid into. If ``None``,
        defaults to ``n_workers``.

    Returns
    -------
    npt.NDArray[np.int32]
        2-D array of shape ``(N, N)`` containing the escape iteration
        count for each pixel.
    """
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size: int = max(1, N // n_chunks)
    chunks: list[tuple] = []
    row: int = 0
    while row < N:
        row_end: int = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # un-timed warm-up: Numba JIT in workers
        parts: list[npt.NDArray[np.int32]] = pool.map(_worker, chunks)
    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    max_iter = 100

    # Serial execution
    start_time: float = time.perf_counter()
    serial_result: npt.NDArray[np.int32] = mandelbrot_serial(
        N, x_min, x_max, y_min, y_max, max_iter
    )
    serial_time: float = time.perf_counter() - start_time
    print(f"Serial execution time: {serial_time:.4f} seconds")

    # Parallel execution
    start_time = time.perf_counter()
    parallel_result: npt.NDArray[np.int32] = mandelbrot_parallel(
        N, x_min, x_max, y_min, y_max, max_iter, n_workers=4, n_chunks=16
    )
    parallel_time: float = time.perf_counter() - start_time
    print(f"Parallel execution time: {parallel_time:.4f} seconds")

    assert np.array_equal(serial_result, parallel_result), "Mismatch!"
    print("Verification passed ✓")

    times: list[float] = []
    for _ in range(3):
        t0: float = time.perf_counter()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial: float = statistics.median(times)

    # Varying number of workers and chunks
    for n_workers in range(1, os.cpu_count() + 1):
        n_chunks = n_workers * 4
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
                n_workers=n_workers,
                n_chunks=n_chunks,
            )
            times.append(time.perf_counter() - t0)
        t_par: float = statistics.median(times)
        speedup: float = t_serial / t_par
        print(
            f"{n_workers:2d} workers: {t_par:.3f}s, "
            f"speedup={speedup:.2f}x, eff={speedup / n_workers * 100:.0f}%"
        )
