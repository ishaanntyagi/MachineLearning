import numpy as np
arr=np.array([[1, 2, 3]])
print(arr)
print(arr.shape)
print(arr.dtype)
print(arr.ndim)
print(arr.size)
print(arr.itemsize)
arr2=np.zeros((2,5))
print(arr2)
arr3=np.full((2,5),7)
print(arr3)
arr4=np.arange(1,10,5)
print(arr4)
arr5 = np.array([[1, 2], [4, 5],[10,11]])
print(arr5)
print(arr5.reshape(2, 3))   # Converts 2x3 → 3x2
print(arr5.flatten())       # Converts 2D → 1D
print(arr5.T)               # Transpose (swap rows/columns)
