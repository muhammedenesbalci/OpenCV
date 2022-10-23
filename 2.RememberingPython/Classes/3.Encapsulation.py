class BankAccount(object):
    def __init__(self, name, money, address):
        self.name = name  # Global Variable
        self.__money = money  # Private Variable(we can not reach or change outside of class)
        self.address = address


p1 = BankAccount("enes", 8000, "Ankara")
print(p1.name)

p1.name = "efe"
print(p1.name)

"""
p1.money #Error
"""


# Attributes are not changeable
class BankAccount2(object):
    def __init__(self, name, money, address):
        self.__name = name
        self.__money = money
        self.__address = address

    def getName(self):
        return self.__name

    def getMoney(self):
        return self.__money

    def getAddress(self):
        return self.__address

    def setName(self, name):
        self.__name = name

    def setMoney(self, money):
        self.__money = money

    def setAddress(self, address):
        self.__address = address


a1 = BankAccount2("Enes", 8000, "Ankara")
print(a1.getName())
a1.setName("Efe")
print(a1.getName())

# Attributes and methods not changeable
class BankAccount3(object):
    def __init__(self, name, money, address):
        self.__name = name
        self.__money = money
        self.__address = address

    def getName(self):
        return self.__name

    def getMoney(self):
        return self.__money

    def getAddress(self):
        return self.__address

    def setName(self, name):
        self.__name = name

    def setMoney(self, money):
        self.__money = money

    def setAddress(self, address):
        self.__address = address

    def __increase(self): # double underscore again the method is private, we can only use it in class
        self.__money = self.__money + 5000


a1 = BankAccount3("Enes", 8000, "Ankara")
a1.increase()
print(a1.getMoney())