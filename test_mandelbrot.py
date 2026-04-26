import numpy as np
import pytest
from mini_project_1.mandelbrot_naive import mandelbrot_naive
from mini_project_1.mandelbrot_numpy import mandelbrot_numpy
from mini_project_1.mandelbrot_naive_numba import mandelbrot_naive_numba as mandelbrot_numba
from mini_project_2.mandelbrot_parallel_chunk_lec5_m3 import mandelbrot_chunk
from mini_project_2.mandelbrot_parallel_chunk_lec5_m3 import mandelbrot_parallel
from mini_project_2.mandelbrot_parallel_chunk_lec5_m3 import mandelbrot_pixel
from mini_project_2.mandelbrot_parallel_chunk_lec5_m3 import mandelbrot_serial
from dask.distributed import Client, LocalCluster

# naive vs numpy
def test_numpy_matches_naive():
    naive = mandelbrot_naive(width=32, height=32, max_iter=20)
    numpy_res = mandelbrot_numpy(width=32, height=32, max_iter=20)

    assert np.array_equal(naive, numpy_res)

# naive vs numba
def test_numba_matches_naive():
    naive = mandelbrot_naive(width=32, height=32, max_iter=20)
    numba_res = mandelbrot_numba(width=32, height=32, max_iter=20)

    assert np.array_equal(naive, numba_res)

# naive vs parallel
def test_parallel_matches_naive():
    N = 32
    args = (-2.0, 1.0, -1.5, 1.5, 20)

    naive = mandelbrot_naive(width=N, height=N, max_iter=20)
    parallel = mandelbrot_parallel(N, *args, n_workers=2, n_chunks=4)
    diff = naive != parallel
    print("Number of differences:", np.sum(diff))
    print("First difference at:", np.argwhere(diff)[:5])
    assert np.array_equal(naive, parallel)

# multiprocessing 
# testing the worker
def test_pixel_inside():
    # c = 0 → always bounded
    val = mandelbrot_pixel(0.0, 0.0, 50)
    assert val == 50

# testing the worker
def test_pixel_outside():
    # far outside → escapes quickly
    val = mandelbrot_pixel(2.0, 2.0, 50)
    assert val < 5

# test chunk function
def test_chunk_values_non_negative():
    res = mandelbrot_chunk(0, 10, 10, -2, 1, -1, 1, 20)
    assert np.all(res >= 0)

# cross-validation
def test_parallel_matches_serial():
    N = 32
    args = (-2.0, 1.0, -1.5, 1.5, 20)

    serial = mandelbrot_serial(N, *args)
    parallel = mandelbrot_parallel(N, *args, n_workers=2, n_chunks=4)

    assert np.array_equal(serial, parallel)

# parameterized test
@pytest.mark.parametrize("N", [16, 32, 64])
def test_parallel_consistency(N):
    args = (-2.0, 1.0, -1.5, 1.5, 20)

    serial = mandelbrot_serial(N, *args)
    parallel = mandelbrot_parallel(N, *args, n_workers=2, n_chunks=4)

    assert np.array_equal(serial, parallel)
    
# dask
def test_dask():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)

    N = 32
    args = (-2.0, 1.0, -1.5, 1.5, 20)

    expected = mandelbrot_chunk(0, N, N, *args)

    future = client.submit(mandelbrot_chunk, 0, N, N, *args)
    result = client.gather(future)

    assert np.array_equal(expected, result)

    client.close()
    cluster.close()