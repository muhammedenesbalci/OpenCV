"""
* Erode
- erodes the boundaries of the foreground object

* Dilation
- The opposite of erosion, magnifies the foreground region

* Opening
- Erode + Dilation
- Reduce the white noise

* CLosing
- Dilation + Erode
- Reduce the balck noises
- close the black points

* Morphology Gradian
- Dilation - Erode
- difference between them
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# transfer img in
img = cv2.imread("img.png",  0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("Original")

# Erode
kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img, kernel, iterations = 2)
plt.figure(), plt.imshow(result, cmap = "gray"), plt.axis("off"), plt.title("Erode")

# Dilation
kernel = np.ones((5,5), dtype = np.uint8)
result2 = cv2.dilate(img, kernel, iterations = 2)
plt.figure(), plt.imshow(result2, cmap = "gray"), plt.axis("off"), plt.title("Dilation")

# White noise
whiteNoise = np.random.randint(0,2, size = img.shape)
whiteNoise = whiteNoise*255
plt.figure(), plt.imshow(whiteNoise, cmap = "gray"), plt.axis("off"), plt.title("White Noise")

noise_img = whiteNoise + img
plt.figure(), plt.imshow(noise_img, cmap = "gray"), plt.axis("off"), plt.title("Img with White Noise")

# Opening
opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Opening")

# Black noise
blackNoise = np.random.randint(0,2, size = img.shape[:2])
blackNoise = blackNoise*-255
plt.figure(), plt.imshow(blackNoise, cmap = "gray"), plt.axis("off"), plt.title("Black Noise")

black_noise_img = blackNoise + img
black_noise_img[black_noise_img <= -245] = 0
plt.figure(), plt.imshow(black_noise_img, cmap = "gray"), plt.axis("off"), plt.title("Black Noise Img")

# Closing
closing = cv2.morphologyEx(black_noise_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Closing")

# Gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure(), plt.imshow(gradient, cmap = "gray"), plt.axis("off"), plt.title("Gradient")

plt.show()




