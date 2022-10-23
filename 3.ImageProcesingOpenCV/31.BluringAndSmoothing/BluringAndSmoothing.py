"""
*Blurring
reducing noise
reduce details
reduce quality

-Mean blurring
think about a box(kernel) 5x5, it have a mean and set it to center pixel

-Gauss Blurring
-Median Blurring
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Opening img
img = cv2.imread("NYC.jpg")
img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img), plt.show()