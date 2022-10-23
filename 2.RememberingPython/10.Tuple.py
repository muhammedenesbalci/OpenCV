print("# Tuple ------------------------------------")

"""
Not changable and sorted a list variation
"""

tuple_variable = (1,2,3,4,5,6,6,6)

print(tuple_variable)

print(tuple_variable[1:6:])

# Counting an variable
print(tuple_variable.count(6))

# We can use tuple to decomposition

x, y, z = (1, 2, 3)
print(x, y, z)