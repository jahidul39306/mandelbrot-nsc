"""
Mandelbrot Set Generator
Author : Md Jahidul Islam Noor
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_point(c, max_iter=100):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return max_iter


def compute_mandelbrot(
    xmin=-2.0, xmax=1.0,
    ymin=-1.5, ymax=1.5,
    width=1024, height=1024,
    max_iter=100
):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=int)

    for i in range(height):
        for k in range(width):
            c = x_vals[k] + 1j * y_vals[i]
            result[i, k] = mandelbrot_point(c, max_iter)

    return result

data = compute_mandelbrot(width=100, height=100, max_iter=50)
print(data.shape)   # (100, 100)
print(data.min(), data.max())
plt.imshow(data, cmap="inferno")
plt.colorbar(label="Iterations")
plt.title("Naive Mandelbrot (100x100)")
plt.show()