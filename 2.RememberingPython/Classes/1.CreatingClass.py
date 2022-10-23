# Class Employee: class can start like thi, too
class Employee(object):
    pass


# Create an object from class
print("Create an object from class--------------------------------")
employee = Employee()


class Footballer:
    age = 37
    club = "Mancherster United"


# Printing attributes
print("Printing attributes------------------------------------")
f1 = Footballer()
print(f1)
print(f1.age)
print(f1.club)

# Changing attributive
print("Changing attributive---------------------------------")
f1.age = 31
print(f1.age)

# Class with attributes and methods
print("Class with attributes and methods---------------------------------------")


class Square:
    edge = 5
    area = 0

    def areaF(self):  # We write "self" in function, and we can use local variables of class
        self.area = self.edge * self.edge
        return self.area


s1 = Square()
s1.edge = 7
print(s1.areaF())

# Methods versus Class
#We say methods that function in a class
print("Methods versus Class-------------------------")
class Emp(object):
    age = 25
    salary = 9500

    def ageSalaryRation(self):
        print(self.age/self.salary)

def ageSalaryRatio(age, salary):
    print(age/salary)

e1 = Emp()
e1.ageSalaryRation()
ageSalaryRatio(15, 10000)

