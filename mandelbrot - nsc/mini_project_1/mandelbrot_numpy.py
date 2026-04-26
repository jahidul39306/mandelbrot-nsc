import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def mandelbrot_numpy(
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    width: int = 1024,
    height: int = 1024,
    max_iter: int = 100,
) -> npt.NDArray[np.int_]:
    """Compute the Mandelbrot set over a 2-D grid using vectorised NumPy operations.

    Constructs a complex-valued meshgrid spanning [``xmin``, ``xmax``] ×
    [``ymin``, ``ymax``] and iterates all pixels simultaneously using NumPy
    masked array updates, avoiding any explicit Python loop over pixels.

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
    npt.NDArray[np.int_]
        2-D array of shape ``(height, width)`` containing the escape
        iteration count for each pixel.
    """
    x: npt.NDArray[np.float64] = np.linspace(xmin, xmax, width)
    y: npt.NDArray[np.float64] = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C: npt.NDArray[np.complex128] = X + 1j * Y

    Z: npt.NDArray[np.complex128] = np.zeros_like(C)
    M: npt.NDArray[np.int_] = np.full(C.shape, max_iter, dtype=int)

    for n in range(max_iter):
        mask: npt.NDArray[np.bool_] = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        escaped: npt.NDArray[np.bool_] = (np.abs(Z) > 2) & (M == max_iter)
        M[escaped] = n + 1
    return M


def benchmark_numpy() -> None:
    """Benchmark :func:`mandelbrot_numpy` across multiple resolutions and display results.

    Runs the NumPy Mandelbrot computation at four increasing resolutions,
    records the elapsed wall-clock time for each, prints a summary to stdout,
    and displays the resulting image for each resolution using matplotlib.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    resolutions: list[tuple[int, int]] = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    times: list[float] = []

    for width, height in resolutions:
        t0: float = time.perf_counter()
        result: npt.NDArray[np.int_] = mandelbrot_numpy(-2, 1, -1.5, 1.5, width, height)
        times.append(time.perf_counter() - t0)

    for (width, height), elapsed in zip(resolutions, times):
        print(f"Resolution: {width}x{height}, Time: {elapsed:.3f} seconds")
        plt.imshow(result, cmap="viridis")
        plt.colorbar(label="Iterations")
        plt.title(f"Mandelbrot Set – viridis colormap, Resolution: {width}x{height}")
        plt.show()


if __name__ == "__main__":
    mandelbrot_numpy()
