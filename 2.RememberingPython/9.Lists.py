print("# List ------------------------------------")

my_list = [1, 2, 3, "Enes", 5.6, 'a']
my_list1 = [[1, 2, 3], ["Muhammed", "enes", "BALCI"]]
my_list2 = [1, 2, 3, 4]
my_list3 = [2, 45, 21, 3412, 31, 1]

print(type(my_list))

print(my_list[3])

# number of elements
print(len(my_list1))

# List in list
print(my_list1[1][0])

# Last element of a list
print(my_list1[-1][-1])

# writing a certain range first index ,including, last index not including, optional skip value
print(my_list[1:4])
print(my_list[0:3:2])

# Adding a new element
my_list2.append(5)
print(my_list2)

# delete an element
my_list2.remove(2)
print(my_list2)

# Reverse list
my_list2.reverse()
print(my_list2)

# Sorting the list
my_list3.sort()
print(my_list3)

# max element
x = max(my_list3)
print(x)

# min element
x = min(my_list3)
print(x)

# addition two list
print(my_list2 + my_list3)


