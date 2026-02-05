"""
Mandelbrot Set Generator
Author : Jahidul
Course : Numerical Scientific Computing 2026
"""

def mandelbrot_point(c, max_iter=100):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return max_iter

# Inside the set → should return max_iter
print(mandelbrot_point(0))          # expected: 100

# Outside the set → should escape fast
print(mandelbrot_point(2 + 2j))     # small number

# Boundary-ish point
print(mandelbrot_point(-0.75 + 0j))