import numpy as np
import time
import matplotlib.pyplot as plt


def mandelbrot_vectorized(xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5, width=1024, height=1024, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    for n in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] = n
    return M

resolutions = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

results = []

for width, height in resolutions:
    start = time.perf_counter()
    result = mandelbrot_vectorized(-2, 1, -1.5, 1.5, width, height)
    elapsed = time.perf_counter() - start
    results.append(elapsed)

for (width, height), elapsed in zip(resolutions, results):
    print(f"Resolution: {width}x{height}, Time: {elapsed:.3f} seconds")
    plt.imshow(result, cmap="viridis")
    plt.colorbar(label="Iterations")
    plt.title(f"Mandelbrot Set – viridis colormap, Resolution: {width}x{height}")
    plt.show()