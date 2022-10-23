print("# Functions------------------------------")

def areaOfCircle(r):
    pi = 3.14

    return pi*(r**2)

print(areaOfCircle(10))


#Paramater have a beggining value but we can change the value
def areaOfCircleWithParameter(r, pi = 3.14):
    return pi*(r**2)

print(areaOfCircleWithParameter(10,5))

# Lambda function for little procces
def mult(x,y,z):
    return x*y*z

print(mult(2,3,4))

# with LAMBDA
funct_mult =lambda x,y,z : x*y*z
print(funct_mult(1,2,3))