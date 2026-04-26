import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numba import njit


@njit
def mandelbrot_numba_typed(
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    width: int = 1024,
    height: int = 1024,
    max_iter: int = 100,
    dtype: Any = np.float64,
) -> npt.NDArray[np.int32]:
    """Compute the Mandelbrot set with a fully JIT-compiled loop at configurable precision.

    Generates a ``height × width`` grid spanning the complex rectangle
    [``xmin``, ``xmax``] × [``ymin``, ``ymax``] and evaluates the Mandelbrot
    escape count for each pixel. The floating-point precision of the coordinate
    arrays is controlled via ``dtype``, allowing direct comparison of
    ``float32`` vs ``float64`` accuracy and performance.

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
    dtype : Any, optional
        NumPy floating-point dtype used for coordinate arrays, e.g.
        ``np.float32`` or ``np.float64``. Default is ``np.float64``.

    Returns
    -------
    npt.NDArray[np.int32]
        2-D array of shape ``(height, width)`` containing the escape
        iteration count for each pixel.
    """
    x_vals = np.linspace(xmin, xmax, width).astype(dtype)
    y_vals = np.linspace(ymin, ymax, height).astype(dtype)

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


def plot_diffrent_precision() -> None:
    """Render and save a side-by-side precision comparison of float32 vs float64.

    Computes the Mandelbrot set at 1024×1024 resolution twice — once with
    ``float32`` coordinates and once with ``float64`` — then plots them
    side by side using a ``hot`` colormap. The maximum absolute pixel
    difference between the two results is printed to stdout and the figure
    is saved to ``precision_comparison.png``.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    r32: npt.NDArray[np.int32] = mandelbrot_numba_typed(
        -2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32
    )
    r64: npt.NDArray[np.int32] = mandelbrot_numba_typed(
        -2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float64
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, result, title in zip(axes, [r32, r64], ["float32", "float64 (ref)"]):
        ax.imshow(result, cmap="hot")
        ax.set_title(title)
        ax.axis("off")

    plt.savefig("precision_comparison.png", dpi=150)
    print(f"Max diff float32 vs float64: {np.abs(r32 - r64).max()}")


if __name__ == "__main__":
    # Warm up JIT for both dtypes
    mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32)
    mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float64)

    for dtype in [np.float32, np.float64]:
        t0: float = time.perf_counter()
        mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=dtype)
        print(f"{dtype.__name__}: {time.perf_counter() - t0:.3f}s")

    plot_diffrent_precision()
