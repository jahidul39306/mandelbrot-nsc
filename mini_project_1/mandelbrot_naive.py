import time

import numpy as np
import numpy.typing as npt


def mandelbrot_point(c: complex, max_iter: int = 100) -> int:
    """Compute the Mandelbrot escape iteration count for a single point.

    Iterates z = z² + c starting from z = 0 and returns the number of
    iterations taken to escape the radius-2 boundary, or ``max_iter`` if
    the point does not escape.

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
    z: complex = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return max_iter


def mandelbrot_naive(
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    width: int = 1024,
    height: int = 1024,
    max_iter: int = 100,
) -> npt.NDArray[np.int_]:
    """Compute the Mandelbrot set over a 2-D grid using pure Python loops.

    Evaluates :func:`mandelbrot_point` for every pixel in a ``height × width``
    grid that spans the complex rectangle [``xmin``, ``xmax``] ×
    [``ymin``, ``ymax``].

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
        Maximum iteration count passed to :func:`mandelbrot_point`.
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
            result[i, k] = mandelbrot_point(c, max_iter)

    return result


def benchmark_naive() -> None:
    """Run :func:`mandelbrot_naive` three times and print elapsed times.

    Executes the naive Mandelbrot computation on a 1024×1024 grid over
    the standard domain three times, printing each run's duration and the
    full list of timings.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    results: list[float] = []
    for _ in range(3):
        start: float = time.perf_counter()
        mandelbrot_naive(-2, 1, -1.5, 1.5, 1024, 1024)
        elapsed: float = time.perf_counter() - start
        print(f"Computation took {elapsed:.3f} seconds")
        results.append(elapsed)
    print(results)


if __name__ == "__main__":
    mandelbrot_naive()
