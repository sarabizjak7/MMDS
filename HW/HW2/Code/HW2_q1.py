import numpy as np
from scipy import linalg

# SVD of M:
M = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
U, Sigma, V_transpose = linalg.svd(M, full_matrices = False)
V = V_transpose.transpose()

sig = np.array([[7.61577311, 0], [0, 1.41421356]])
print("----- U -----")
print(U)
print("----- Sigma -----")
print(Sigma)
print("----- V_transpose -----")
print(V_transpose)
print("----- V -----")
print(V)

# Eigenvalues and eigen vectors of M_transpose x M:
M_transpose = M.transpose()
mult = M_transpose.dot(M)
#print(mult)
Evals, Evecs = linalg.eigh(mult)

print("----------")
print(Evals)
print("----------")
print(Evecs)
print("----------")


print("-------- Evals --------")
print(Evals)
print("-------- Evecs --------")
print(Evecs)


"""
# Test ortogonalnosti : 
ET = Evecs.transpose()
print(" PRODUKT ")
print(np.dot(V, V_transpose))
print(np.dot(V_transpose, V))
print("_______________________")
print(np.dot(Evecs,ET))
print(np.dot(ET, Evecs))


# Check: 
print(np.dot(np.dot(U, sig), V_transpose))
"""