import sys

import numpy
import matplotlib
import scipy
import numba
import pytest
import dask
import distributed


def main():
    print("Python version:", sys.version)
    print("-" * 40)

    print("numpy:", numpy.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("scipy:", scipy.__version__)
    print("numba:", numba.__version__)
    print("pytest:", pytest.__version__)
    print("dask:", dask.__version__)
    print("distributed:", distributed.__version__)

    print("-" * 40)
    print("✅ All packages imported successfully!")


if __name__ == "__main__":
    main()
