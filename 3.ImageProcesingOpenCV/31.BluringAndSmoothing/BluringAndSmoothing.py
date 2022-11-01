"""
*Blurring
reducing noise
reduce details
reduce quality

-Mean blurring
think about a box(kernel) 5x5, it have a mean and set it to center pixel

-Gauss Blurring
same but with, Gauissan formulas

-Median Blurring
same but takes median in our box(kernel)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Opening img
img = cv2.imread("NYC.jpg")
img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img)

# Mean Blurring
mean_blur = cv2.blur(img, ksize=(3, 3))  # ksize(kernel size) = our matrix dimension
plt.figure(), plt.imshow(mean_blur), plt.title("Mean Bluriing")

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=7)
plt.figure(), plt.imshow(gaussian_blur), plt.title("Gauss Blur")

# Median Blur
mb = cv2.medianBlur(img, ksize=3)
plt.figure(), plt.imshow(mb), plt.axis("off"), plt.title("Median Blur")

# Adding Noise Function
def gaussianNoise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var ** 0.5  # variance

    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss  #  Images are arrays so we can sum of them.

    return noisy


# transfer in
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255  # Normalizing values of image
plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("original normalized")

# Create image with Gaussian noise
gaussianNoisyImage = gaussianNoise(img)
plt.figure(), plt.imshow(gaussianNoisyImage), plt.axis("off"), plt.title("Image Gaussian Noisy")

# Gaussian Blur on gaussian noise image
gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize=(3, 3), sigmaX=7)
plt.figure(), plt.imshow(gb2), plt.axis("off"), plt.title("Image Gaussian Noisy with Gauss Blur")

# Adding salt and pepper noise function
def saltPepperNoise(image):
    row, col, ch = image.shape
    saltAndPepperRatio = 0.5

    amount = 0.004

    noisy = np.copy(image)

    # Salt - white
    num_salt = np.ceil(amount * image.size * saltAndPepperRatio)  # ceil do rounding 1.9 -> 2.0
    Coordiantes = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[Coordiantes] = 1

    # Pepper- Black
    num_pepper = np.ceil(amount * image.size * (1 - saltAndPepperRatio))
    Coordiantes = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[Coordiantes] = 0

    return noisy

# Noisy Image
spImage = saltPepperNoise(img)
plt.figure(), plt.imshow(spImage), plt.axis("off"), plt.title("SP Image")

# Applying median blur
mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize=3)
plt.figure(), plt.imshow(mb2), plt.axis("off"), plt.title("SP Image with Median Blur")

plt.show()