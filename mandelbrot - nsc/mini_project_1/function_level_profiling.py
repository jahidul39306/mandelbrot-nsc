import cProfile, pstats
from mandelbrot_naive import mandelbrot_naive
from mandelbrot_numpy import mandelbrot_numpy

cProfile.run("mandelbrot_naive(-2 , 1, -1.5 , 1.5 , 512 , 512)", "naive_profile.prof")
cProfile.run("mandelbrot_numpy(-2 , 1, -1.5 , 1.5 , 512 , 512)", "numpy_profile.prof")

for name in ("naive_profile.prof", "numpy_profile.prof"):
    stats = pstats.Stats(name)
    stats.sort_stats("cumulative")
    stats.print_stats(10)