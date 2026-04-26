import numpy as np
import time

N = 10000
A = np.random.rand(N, N)
A_f = np.asfortranarray(A)

def sum_rows(A):
    for i in range(A.shape[0]):
        _ = np.sum(A[i, :])

def sum_cols(A):
    for j in range(A.shape[1]):
        _ = np.sum(A[:, j])

start = time.perf_counter()
sum_rows(A)
elapsed_rows = time.perf_counter() - start  
print(f"Row-wise sum took {elapsed_rows:.3f} seconds")
start = time.perf_counter()
sum_cols(A)
elapsed_cols = time.perf_counter() - start  
print(f"Column-wise sum took {elapsed_cols:.3f} seconds")

start = time.perf_counter()
sum_rows(A_f)
elapsed_rows_f = time.perf_counter() - start  
print(f"Row-wise sum (Fortran order) took {elapsed_rows_f:.3f} seconds")
start = time.perf_counter()
sum_cols(A_f)  
elapsed_cols_f = time.perf_counter() - start  
print(f"Column-wise sum (Fortran order) took {elapsed_cols_f:.3f} seconds")