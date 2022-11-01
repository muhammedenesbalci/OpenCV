import numpy as np

print("matrix--------------------------------")
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9 ,10], [11, 12, 13, 14, 15]])
print(arr)

print("Shape of matrix------------")
print(arr.shape)

print("Cropped matrix think as a figure")
print(arr[0:2, 3:])  # as (row and column) (y, x)


# but opencv excpet like this [3:, 0:2] hope you are understand
"""
points (Column, Row) (X, Y)
0/0-----X----->
 |
 |
 Y
 |
 |
 v
"""