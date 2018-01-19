from numpy import *

arr = [
    [1, 2],
    [1, 3],
    [2, 3],
    [10, 3]
]
mat = mat(arr)

print(mat.shape[0])  # row
print(mat.shape[1])  # col

mat = zeros((1,1024))
print(mat)