import matplotlib.pyplot as plt
import numpy as np

labels = [
    "Naive", "NumPy", "Hybrid", "Numba",
    "Multiprocessing", "Dask local", "Dask cluster",
    "GPU f32", "CPU f64"
]

t1024 = [5.598, 0.586, 1.858, 0.075, 0.383, 0.106, 0.099, 0.0018, 0.0167]
t4096 = [88.776, 18.237, 29.740, 1.243, 3.997, 1.305, 1.377, 0.0187, 0.2432]

base_1024 = t1024[0]
base_4096 = t4096[0]

speedup_1024 = [base_1024 / t for t in t1024]
speedup_4096 = [base_4096 / t for t in t4096]

x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(13, 6))

bars1 = ax.bar(x - w/2, speedup_1024, width=w, label="1024×1024", color="#185FA5", zorder=3)
bars2 = ax.bar(x + w/2, speedup_4096, width=w, label="4096×4096", color="#1D9E75", zorder=3)

# Log scale — makes all bars readable
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=11)
ax.set_ylabel("Speedup (×) — log scale", fontsize=12)
ax.set_title("Mandelbrot benchmark speedup comparison", fontsize=14)
ax.legend(fontsize=11)
ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Value labels on top of each bar
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15,
            f"{h:.1f}×", ha="center", va="bottom", fontsize=7.5, color="#185FA5")

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15,
            f"{h:.1f}×", ha="center", va="bottom", fontsize=7.5, color="#1D9E75")

plt.tight_layout()
plt.savefig("mandelbrot_speedup.png", dpi=150, bbox_inches="tight")
plt.show()