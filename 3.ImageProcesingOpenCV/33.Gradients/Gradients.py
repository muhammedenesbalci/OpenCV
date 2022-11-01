"""
* Used for edge detection
* According to color intensities
"""

import cv2
import matplotlib.pyplot as plt

# Transfer img in
img = cv2.imread("sudoku.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off"), plt.title("Original")

# X axis Gradian
sobelX = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)  # ddept output depth
plt.figure(), plt.imshow(sobelX, cmap="gray"), plt.axis("off"), plt.title("Gradian X")

# Y axis Gradian
sobelY = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobelY, cmap="gray"), plt.axis("off"), plt.title("Gradian Y")

# X + Y weird it does not work
sobelXY = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobelXY, cmap="gray"), plt.axis("off"), plt.title("Gradian XY")

# Laplacian Gradient
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S, ksize = 5)
plt.figure(), plt.imshow(laplacian, cmap="gray"), plt.axis("off"), plt.title("Laplacian")

plt.show()