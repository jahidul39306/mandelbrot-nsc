# Mandelbrot Set Performance Benchmarking

**Author:** Md Jahidul Islam Noor  
**Course:** Numerical Scientific Computing (NSC) — Aalborg University  
**Environment:** `environment.yml` (see setup below)

---

## Overview

This project benchmarks nine different implementations of the Mandelbrot set computation, ranging from a pure Python naive loop to GPU-accelerated OpenCL kernels. The goal is to understand how different parallelisation and optimisation strategies affect performance at two grid resolutions: **1024×1024** and **4096×4096**.

---

## Implementations

| # | Implementation | Description |
|---|---------------|-------------|
| 1 | Naive | Pure Python nested loops |
| 2 | NumPy | Vectorised operations with NumPy |
| 3 | Hybrid | NumPy + partial Python loop |
| 4 | Numba | JIT-compiled with `@njit` |
| 5 | Multiprocessing | CPU parallelism via `multiprocessing.Pool` |
| 6 | Dask local | Lazy parallel chunks on local machine |
| 7 | Dask cluster | Distributed Dask with local cluster |
| 8 | GPU f32 | OpenCL GPU kernel, 32-bit float |
| 9 | GPU f64 | OpenCL GPU kernel, 64-bit float (CPU device fallback) |

---

## Results

### Execution time

| Implementation | 1024×1024 (s) | 4096×4096 (s) |
|---------------|--------------|--------------|
| Naive | 5.598 | 88.776 |
| NumPy | 0.586 | 18.237 |
| Hybrid | 1.858 | 29.740 |
| Numba | 0.075 | 1.243 |
| Multiprocessing | 0.383 | 3.997 |
| Dask local | 0.106 | 1.305 |
| Dask cluster | 0.099 | 1.377 |
| GPU f32 | 0.0018 | 0.0187 |
| GPU f64 | 0.0167 | 0.2432 |

### Speedup over naive baseline

| Implementation | 1024×1024 | 4096×4096 |
|---------------|-----------|-----------|
| NumPy | 9.6× | 4.9× |
| Hybrid | 3.0× | 3.0× |
| Numba | 74.6× | 71.4× |
| Multiprocessing | 14.6× | 22.2× |
| Dask local | 52.8× | 68.0× |
| Dask cluster | 56.5× | 64.5× |
| GPU f32 | 3,110× | 4,748× |
| GPU f64 | 335× | 365× |

### Key findings

- **GPU f32 is the fastest** implementation by a large margin — over 3,000× faster than naive at 1024×1024 and nearly 5,000× at 4096×4096.
- **GPU f64** requires falling back to a CPU OpenCL device on Intel UHD Graphics (which does not support `cl_khr_fp64`), which reduces its advantage to ~335–365×, but it still far outperforms all pure CPU approaches.
- **Numba** is the best CPU-only option, achieving ~75× speedup through JIT compilation with minimal code changes.
- **Dask** (both local and cluster) performs comparably to Numba at large grid sizes, showing good scalability.
- **NumPy** provides a solid baseline improvement (~5–10×) with no extra dependencies beyond the standard scientific stack.
- **Multiprocessing** scales better at larger grid sizes (22× at 4096 vs 15× at 1024) due to better amortisation of process spawn overhead.

---

## Project structure

```
MANDELBROT - NSC/
├── mini_project_1/
├── mini_project_2/
├── mini_project_3/
├── .coverage
├── .gitignore
├── benchmark_results.txt
├── benchmark.py
├── dask_chunk_sweep.png
├── dask_local_chunk_sweep.png
├── environment.yml
├── git_log.txt
├── speedup_bar_chart.png
└── test_mandelbrot.py
```

---

## Setup and usage

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate mp3
```

### 2. Run any implementation directly

```bash
python [file_name].py
```

### 3. Run tests

```bash
pytest
```

---

## Notes on GPU f64

Intel integrated GPUs (e.g. Intel UHD Graphics) do not support the `cl_khr_fp64` OpenCL extension required for native double-precision computation. The `mandelbrot_gpu_f64.py` implementation therefore targets the **CPU OpenCL device**, which does support float64. This is selected automatically at runtime.