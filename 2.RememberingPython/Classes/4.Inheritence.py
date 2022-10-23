print("Animals-----------------------------------------------")


# Parent
class Animal(object):
    def __init__(self):
        print("Animal created")

    def atr(self):
        print("Animal")

    def walk(self):
        print("walking")


# Child
class Monkey(Animal):
    def __init__(self):
        super().__init__()  # We pass the constructor of the main class
        #Animal.__init__(self)
        print("Monkey created")

    def atr(self):  # Overriding
        print("Monkey")

    def climb(self):
        print("Monkey can climb")


monk = Monkey()
monk.atr()
monk.walk()

print("Website-----------------------------------------------")


# Parent
class Website:
    def __init__(self, name, surname):
        self.name = name
        self.surname = surname

    def loginInfo(self):
        print(self.name + " " + self.surname)


# Child

class Website2(Website):
    def __init__(self, name, surname, ids):
        Website.__init__(self, name, surname)  # First constructor have an initial values, so we have to use like this
        #super().__init__(name, surname)
        self.ids = ids

    def login(self):
        print(self.name + " " + self.surname + " " + self.ids)

w1 = Website("enes", "balcı")
w2 = Website2("Enes", "balcı", 31)


print(w2.name)