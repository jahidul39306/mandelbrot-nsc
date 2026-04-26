import statistics
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def mandelbrot_point_numba(c: complex, max_iter: int = 100) -> int:
    """Compute the Mandelbrot escape iteration count for a single point (JIT-compiled).

    Iterates z = z² + c starting from z = 0 and returns the number of
    iterations taken to escape the radius-2 boundary, or ``max_iter`` if
    the point does not escape. Uses real/imag component checks to avoid
    ``abs()`` overhead inside the Numba-compiled loop.

    Parameters
    ----------
    c : complex
        The complex coordinate to test.
    max_iter : int, optional
        Maximum number of iterations before declaring the point inside
        the Mandelbrot set. Default is 100.

    Returns
    -------
    int
        Number of iterations before escape, or ``max_iter`` if the point
        did not escape.
    """
    z = 0j
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z**2 + c
    return max_iter


def mandelbrot_hybrid(
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    width: int = 1024,
    height: int = 1024,
    max_iter: int = 100,
) -> npt.NDArray[np.int_]:
    """Compute the Mandelbrot set using a Python loop with a JIT-compiled inner call.

    Iterates over every pixel in a ``height × width`` grid using a plain
    Python loop, but delegates the per-point escape computation to the
    Numba-compiled :func:`mandelbrot_point_numba`. This hybrid approach
    avoids Python overhead inside the critical path while keeping the outer
    loop in interpreted Python.

    Parameters
    ----------
    xmin : float, optional
        Left boundary of the real axis. Default is -2.0.
    xmax : float, optional
        Right boundary of the real axis. Default is 1.0.
    ymin : float, optional
        Bottom boundary of the imaginary axis. Default is -1.5.
    ymax : float, optional
        Top boundary of the imaginary axis. Default is 1.5.
    width : int, optional
        Number of pixels along the real (horizontal) axis. Default is 1024.
    height : int, optional
        Number of pixels along the imaginary (vertical) axis. Default is 1024.
    max_iter : int, optional
        Maximum iteration count passed to :func:`mandelbrot_point_numba`.
        Default is 100.

    Returns
    -------
    npt.NDArray[np.int_]
        2-D array of shape ``(height, width)`` containing the escape
        iteration count for each pixel.
    """
    x_vals: npt.NDArray[np.float64] = np.linspace(xmin, xmax, width)
    y_vals: npt.NDArray[np.float64] = np.linspace(ymin, ymax, height)

    result: npt.NDArray[np.int_] = np.zeros((height, width), dtype=int)

    for i in range(height):
        for k in range(width):
            c: complex = x_vals[k] + 1j * y_vals[i]
            result[i, k] = mandelbrot_point_numba(c, max_iter)

    return result


@njit
def mandelbrot_naive_numba(
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    width: int = 1024,
    height: int = 1024,
    max_iter: int = 100,
) -> npt.NDArray[np.int32]:
    """Compute the Mandelbrot set with a fully JIT-compiled nested loop (Numba).

    Both the outer pixel loop and the inner iteration loop are compiled by
    Numba's ``@njit`` decorator, eliminating all Python interpreter overhead.
    The escape condition is evaluated via squared real/imaginary components
    to avoid the cost of ``abs()``.

    Parameters
    ----------
    xmin : float, optional
        Left boundary of the real axis. Default is -2.0.
    xmax : float, optional
        Right boundary of the real axis. Default is 1.0.
    ymin : float, optional
        Bottom boundary of the imaginary axis. Default is -1.5.
    ymax : float, optional
        Top boundary of the imaginary axis. Default is 1.5.
    width : int, optional
        Number of pixels along the real (horizontal) axis. Default is 1024.
    height : int, optional
        Number of pixels along the imaginary (vertical) axis. Default is 1024.
    max_iter : int, optional
        Maximum number of Mandelbrot iterations per pixel. Default is 100.

    Returns
    -------
    npt.NDArray[np.int32]
        2-D array of shape ``(height, width)`` containing the escape
        iteration count for each pixel.
    """
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


def bench(fn: Callable[..., object], *args: object, runs: int = 5) -> float:
    """Benchmark a callable by running it multiple times and returning the median time.

    Performs one extra warm-up call before timing to account for JIT
    compilation or caching effects, then records ``runs`` timed executions
    and returns their median elapsed time in seconds.

    Parameters
    ----------
    fn : Callable[..., object]
        The function to benchmark.
    *args : object
        Positional arguments forwarded to ``fn`` on every call.
    runs : int, optional
        Number of timed repetitions. Default is 5.

    Returns
    -------
    float
        Median wall-clock time in seconds across all timed runs.
    """
    fn(*args)  # extra warm-up
    times: list[float] = []
    for _ in range(runs):
        t0: float = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


if __name__ == "__main__":
    # Warm up the JIT compiler
    _ = mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64)
    _ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)

    t_hybrid = bench(mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024)
    t_full = bench(mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024)

    print(f" Hybrid        : {t_hybrid:.3f}s")
    print(f" Fully compiled: {t_full:.3f}s")
    print(f" Ratio         : {t_hybrid / t_full:.1f}x")
