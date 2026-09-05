import fluxioncore.matrix as m
from fluxioncore import DataType

vals = [1, 2, 3, 2, 4, 5, 3, 5, 6]
M = m.Matrix.random_normal(3, 3, 0, 1, DataType.REAL)
N = m.Matrix.random_normal(3, 3, 0, 1, DataType.REAL)
R = M @ N
R.print()


print("shape:")
print(M.shape)
