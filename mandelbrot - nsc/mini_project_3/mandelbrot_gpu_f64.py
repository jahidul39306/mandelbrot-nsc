import pyopencl as cl
import numpy as np
import time, matplotlib.pyplot as plt

KERNEL_SRC = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    // Prevent out-of-bounds
    if (col >= N || row >= N) return;
    
    double c_real = x_min + col * (x_max - x_min) / (double)(N - 1);
    double c_imag = y_min + row * (y_max - y_min) / (double)(N - 1);  
    
    // Mandelbrot iteration
    double zr = 0.0, zi = 0.0;
    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

# --- Explicitly pick a CPU OpenCL device ---
cpu_device = None
for platform in cl.get_platforms():
    for device in platform.get_devices():
        if device.type == cl.device_type.CPU:
            cpu_device = device
            break
    if cpu_device:
        break

if cpu_device is None:
    raise RuntimeError("No CPU OpenCL device found!")
print(f"Using device: {cpu_device.name}")

ctx = cl.Context([cpu_device])
queue = cl.CommandQueue(ctx)
prog = cl.Program(ctx, KERNEL_SRC).build()

# Parameters
N, MAX_ITER = 1024, 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

# Memory
image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)


# --- Warm up (first launch triggers a kernel compile) ---
prog.mandelbrot(
    queue,
    (64, 64),
    None,
    image_dev,
    np.float64(X_MIN),
    np.float64(X_MAX),
    np.float64(Y_MIN),
    np.float64(Y_MAX),
    np.int32(64),
    np.int32(MAX_ITER),
)
queue.finish()

# --- Time the real run ---
t0 = time.perf_counter()
prog.mandelbrot(
    queue,
    (N, N),
    None,
    image_dev,
    np.float64(X_MIN),
    np.float64(X_MAX),
    np.float64(Y_MIN),
    np.float64(Y_MAX),
    np.int32(N),
    np.int32(MAX_ITER),
)
queue.finish()
elapsed = time.perf_counter() - t0

cl.enqueue_copy(queue, image, image_dev)
queue.finish()

print(f"CPU {N}x{N}: {elapsed*1e3:.1f} ms")

plt.imshow(image, cmap="hot", origin="lower")
plt.axis("off")
plt.savefig("mandelbrot_cpu_f64.png", dpi=150, bbox_inches="tight")
