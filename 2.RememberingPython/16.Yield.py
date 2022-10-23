print("# Yield------------------------------")
"""
-iterations
-generations
-yield
"""

# iterator
list = [1, 2, 3, 4]

for i in list:
    print(i)

# Generator
"""
Does not keep it in memory constantly, produces and uses when necessary
"""

generator = (x for x in range(1, 10))
for i in generator:
    print(i)

# Yields
"""
if a function return a generator then we have to use yield key instead of return 
"""

def create_generator():
    list = range(1,5)

    for i in list:
        yield i

generator = create_generator()
print(generator)

for i in generator:
    print(i)