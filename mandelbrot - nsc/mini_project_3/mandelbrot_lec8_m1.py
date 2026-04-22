import numpy as np
import matplotlib.pyplot as plt

N, MAX_ITER, TAU = 512, 1000, 0.01

x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)

C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

z32 = np.zeros_like(C32)
z64 = np.zeros_like(C64)

diverge = np.full((N, N), MAX_ITER, dtype=np.int32)
active = np.ones((N, N), dtype=bool)

for k in range(MAX_ITER):
    if not active.any():
        break

    z32[active] = z32[active] ** 2 + C32[active]
    z64[active] = z64[active] ** 2 + C64[active]

    diff = np.abs(z32.real.astype(np.float64) - z64.real) + np.abs(
        z32.imag.astype(np.float64) - z64.imag
    )

    newly = active & (diff > TAU)

    diverge[newly] = k
    active[newly] = False

# Compute escape-count map for comparison, question: 3
escape = np.full((N, N), MAX_ITER, dtype=np.int32)
z = np.zeros_like(C64)
active2 = np.ones((N, N), dtype=bool)

for k in range(MAX_ITER):
    if not active2.any():
        break

    z[active2] = z[active2] ** 2 + C64[active2]
    escaped = active2 & (np.abs(z) > 2.0)
    escape[escaped] = k
    active2[escaped] = False


# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(
    diverge, cmap="plasma", origin="lower", extent=[-0.7530, -0.7490, 0.0990, 0.1030]
)
axes[0].set_title(f"Trajectory divergence (τ={TAU})")
plt.colorbar(axes[0].images[0], ax=axes[0], label="Divergence iteration")

axes[1].imshow(
    escape, cmap="inferno", origin="lower", extent=[-0.7530, -0.7490, 0.0990, 0.1030]
)
axes[1].set_title("Escape-count map")
plt.colorbar(axes[1].images[0], ax=axes[1], label="Escape iteration")

plt.tight_layout()
plt.show()

# Correlation check
mask = escape < MAX_ITER  # only escaped pixels
correlation = np.corrcoef(diverge[mask], escape[mask])[0, 1]
print(f"Correlation (escaped pixels only): {correlation:.4f}")

# question: 1
fraction = np.mean(diverge < MAX_ITER)
print(fraction)
