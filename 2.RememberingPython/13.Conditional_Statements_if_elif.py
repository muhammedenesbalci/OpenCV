print("# if - elif - else------------------------------")

list = [1, 2, 3, 4, 5]

deger = input("Enter number : ")
deger = int(deger)

if deger in list:
    print("Yes")
else:
    print("No")

tur = {"one": 1, "two": 2, "three": 3}

letter = input("Enter str")
letter = str(letter)

if letter in tur.keys():
    print("Yes")
else:
    print("No")
