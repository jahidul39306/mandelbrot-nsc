import numpy as np, time
from numba import njit
import matplotlib.pyplot as plt


@njit
def mandelbrot_numba_typed(
    xmin=-2.0,
    xmax=1.0,
    ymin=-1.5,
    ymax=1.5,
    width=1024,
    height=1024,
    max_iter=100,
    dtype=np.float64,
):
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


def plot_diffrent_precision():
    r32 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float32)
    r64 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, result, title in zip(axes, [r32, r64], ["float32", "float64(ref)"]):
        ax.imshow(result, cmap="hot")
        ax.set_title(title)
        ax.axis("off")
    plt.savefig("precision_comparison.png", dpi=150)
    print(f" Max diff float32 vs float64 : {np.abs(r32 - r64 ). max ()}")


if __name__ == "__main__":
    for dtype in [np.float32, np.float64]:
        t0 = time.perf_counter()
        mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype=dtype)
        print(f"{ dtype.__name__ }: { time.perf_counter()-t0:.3f}s")
    
    #plot_diffrent_precision()
