import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Constants
N, MAX_ITER = 512, 1000
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)
C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)

# Floating point precision settings
eps32 = float(np.finfo(np.float32).eps)
delta = np.maximum(eps32 * np.abs(C), 1e-10)

def escape_count(C, max_iter):
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    
    for k in range(max_iter):
        mask = ~esc
        z[mask] = z[mask]**2 + C[mask]
        newly = mask & (np.abs(z) > 2.0)
        cnt[newly] = k
        esc[newly] = True
    return cnt

# Calculate base and perturbed counts
n_base = escape_count(C, MAX_ITER).astype(float)
n_perturb = escape_count(C + delta, MAX_ITER).astype(float)

# Calculate condition number kappa
dn = np.abs(n_base - n_perturb)
kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)

# Plotting
cmap_k = plt.cm.hot.copy()
cmap_k.set_bad('0.25')
vmax = np.nanpercentile(kappa, 99)

plt.figure(figsize=(10, 8))
plt.imshow(
    kappa, 
    cmap=cmap_k, 
    origin='lower',
    extent=[-0.7530, -0.7490, 0.0990, 0.1030],
    norm=LogNorm(vmin=1, vmax=vmax)
)

plt.colorbar(label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$)')
plt.title(r'Condition number approx $\kappa(c) = |\Delta n|\,/\,(\varepsilon_{32}\,n(c))$')
plt.show()
