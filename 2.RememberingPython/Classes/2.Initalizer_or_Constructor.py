class Animal(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def getAge(self):
        return self.age

    def getName(self):
        return self.name

a1 = Animal("dog", 2)
print(a1.getAge())
print(a1.getName())

a2 = Animal("cat", 5)
print(a2.getAge())
print(a2.getName())
