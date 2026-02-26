import cProfile, pstats
from mandelbrot_naive import mandelbrot_naive
from mandelbrot_numpy import mandelbrot_numpy

cProfile.run("mandelbrot_naive()", "naive_profile.prof")
cProfile.run("mandelbrot_vectorize()", "vectorize_profile.prof")
