import numpy as np

x = np.linspace(-2, 1, 1024)
y = np.linspace(-1.5, 1.5, 1024)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

print (f" Shape : {C. shape }")
print (f" Type : {C. dtype }")