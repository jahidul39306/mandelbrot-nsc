from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """Compute the Mandelbrot escape iteration count for a single point.

    Iterates the Mandelbrot recurrence relation for a given complex
    coordinate and returns the iteration at which the magnitude exceeds
    2, or ``max_iter`` if the point does not escape.

    Parameters
    ----------
    c_real : float
        Real component of the complex number.
    c_imag : float
        Imaginary component of the complex number.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Number of iterations before escape, or ``max_iter`` if bounded.
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
    """Compute a horizontal strip of the Mandelbrot set.

    Evaluates the Mandelbrot escape iteration count for a contiguous
    range of rows in the grid.

    Parameters
    ----------
    row_start : int
        Starting row index (inclusive).
    row_end : int
        Ending row index (exclusive).
    N : int
        Grid size (number of columns and total rows).
    x_min : float
        Minimum value of the real axis.
    x_max : float
        Maximum value of the real axis.
    y_min : float
        Minimum value of the imaginary axis.
    y_max : float
        Maximum value of the imaginary axis.
    max_iter : int
        Maximum number of iterations per point.

    Returns
    -------
    npt.NDArray[np.int32]
        2D array of shape ``(row_end - row_start, N)`` containing
        iteration counts for the specified rows.
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
    """Compute the Mandelbrot set using a serial approach.

    Parameters
    ----------
    N : int
        Grid size (output will be ``N x N``).
    x_min : float
        Minimum value of the real axis.
    x_max : float
        Maximum value of the real axis.
    y_min : float
        Minimum value of the imaginary axis.
    y_max : float
        Maximum value of the imaginary axis.
    max_iter : int, optional
        Maximum number of iterations per point. Default is 100.

    Returns
    -------
    npt.NDArray[np.int32]
        2D array of shape ``(N, N)`` containing iteration counts.
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(
    args: Tuple[int, int, int, float, float, float, float, int],
) -> npt.NDArray[np.int32]:
    """Worker function for parallel Mandelbrot computation.

    Unpacks arguments and computes a chunk of the Mandelbrot set.

    Parameters
    ----------
    args : tuple
        Tuple containing arguments for ``mandelbrot_chunk``.

    Returns
    -------
    npt.NDArray[np.int32]
        Computed chunk of the Mandelbrot set.
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
) -> npt.NDArray[np.int32]:
    """Compute the Mandelbrot set using multiprocessing.

    Splits the grid into horizontal chunks and distributes them across
    multiple worker processes using a process pool.

    Parameters
    ----------
    N : int
        Grid size (output will be ``N x N``).
    x_min : float
        Minimum value of the real axis.
    x_max : float
        Maximum value of the real axis.
    y_min : float
        Minimum value of the imaginary axis.
    y_max : float
        Maximum value of the imaginary axis.
    max_iter : int, optional
        Maximum number of iterations per point. Default is 100.
    n_workers : int, optional
        Number of worker processes. Default is 4.

    Returns
    -------
    npt.NDArray[np.int32]
        2D array of shape ``(N, N)`` containing iteration counts.
    """
    chunk_size: int = max(1, N // n_workers)
    chunks: List[Tuple[int, int, int, float, float, float, float, int]] = []
    row: int = 0

    while row < N:
        row_end: int = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # Warm-up for Numba JIT
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)
