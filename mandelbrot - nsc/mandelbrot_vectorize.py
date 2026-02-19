import numpy as np
import time

x = np.linspace(-2, 1, 1024)
y = np.linspace(-1.5, 1.5, 1024)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

Z = np.zeros_like(C)
M = np.zeros(C.shape, dtype=int)
max_iter = 100

start = time.perf_counter()
for n in range(max_iter):
    mask = np.abs(Z) <= 2
    Z[mask] = Z[mask] ** 2 + C[mask]
    M[mask] = n
elapsed = time.perf_counter() - start
print(f"Computation took {elapsed:.3f} seconds")