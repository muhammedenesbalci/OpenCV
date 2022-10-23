print("# Matplotlib------------------------------")

import matplotlib.pyplot as plt
import numpy as np

# Visualizing
print("Visualizing--------------------------------")
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

plt.figure()
# Line
plt.plot(x, y, color="red", alpha=0.5, label="line")

# Dots
plt.scatter(x, y, color="blue", alpha=0.7, label="dots")

# Label infos
plt.legend()

# give name axis
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fig-1")

# Grids
plt.grid(True)

plt.show()

# One more table
# row,column,size
x = np.array(range(0, 11))
y = x[::-1]

print(x)
print(y)

fig, axex = plt.subplots(2, 1, figsize=(9, 7))

# Spaces between figrues
fig.subplots_adjust(hspace=0.5)

#setting figures
axex[0].scatter(x,y)
axex[0].set_title("sub1")
axex[0].set_xlabel("x1")
axex[0].set_ylabel("y1")

axex[1].plot(y,x)
axex[1].set_title("sub2")
axex[1].set_xlabel("x2")
axex[1].set_ylabel("y2")

plt.show()

# Random Image

plt.figure()
img = np.random.random((10,10))
plt.imshow(img, cmap="gray")
plt.show()
