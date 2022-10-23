print("# Numpy------------------------------")
import numpy as np

# 1*9 Array list
print("1*9 Array list")
list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(list)

list = np.array(range(1, 10))
print(list)

# Shape of an array
print("Shape of an array")
print(list.shape)

# Dimension of array
print("Dimension of array")
print(list.ndim)

# Variable type
print("Variable type")
print(list.dtype.name)

# Size of an array
print("Size of an array")
print(list.size)

# Reshape an array
print("Reshape an array")
list = list.reshape(3, 3)
print(list)

# Array type
print("Array type")
print(type(list))

# 2D array
print("2D array")
array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(array_2d)

# Zeros matrix
print("Zeros matrix")
zeros_array = np.zeros((5, 8))
print(zeros_array)

# Ones array
print("Ones array")
ones_array = np.ones((4, 3))
print(ones_array)
print(ones_array.dtype.name)

# Empty array(near zero values)
print("Empty array(near zero values)")
empty_arr = np.empty((5, 5))
print(empty_arr)

# Arrange (x, y, step)
print("Arrange (x, y, step)")
arrange_arr = np.arange(0, 50, 2)
print(arrange_arr)

# Lin-space (x, y, step) last index include this time
print("Lin-space (x, y, step) last index include this time")
linspace_arr = np.linspace(10, 20, 11)
print(linspace_arr)

# Math operations
print("Math operations")
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])

print(a + b)
print(a - b)
print(a * b)
print(a / b)

# Max-Min
print("Max-Min")
print(np.max(a))
print(np.min(b))

# Random array 0-1
print("Random array")
rand_arr = np.random.random((5, 5))
print(rand_arr)

# Indexes similar to lists
print("Indexes")
array_indx = np.random.random((6, 8))

print(array_indx[2])
print((array_indx[2][2]))  # row-columns
print(array_indx[2:5:2])
print(array_indx[0, 1:4])  # first row and range 1-4(row, range)

# Convert to vector
print("Convert to vector")
arr = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
print(arr)

arr = arr.ravel()
print(arr)

# Index of max-min argument
print("Index of max argument")
arr = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
print(arr.argmax())
print(arr.argmin())
