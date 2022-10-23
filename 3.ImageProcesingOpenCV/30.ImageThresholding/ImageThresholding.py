import cv2
import matplotlib.pyplot as plt

# Transfer in
img = cv2.imread("img1.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("ImageThresh", img)
print(img.shape)

# Show with matplotlib
plt.figure()
plt.imshow(img, cmap = "gray")  # BGR-GRAY so we have to use cmap = "gray"
plt.show()

# Thresholding Binary
_, thresh_img = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY)  # thresh min amplitude maxval max amplitude, images have 0-255, (60-255 white for Thresh_Bınary)(60-255 BLACCK THRESH BINATY_IN)
plt.figure()
plt.imshow(thresh_img, cmap = "gray")
plt.show()

# Thresholding Binary-ınverse
_, thresh_img = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY_INV  )  # thresh min amplitude maxval max amplitude, images have 0-255, (60-255 white for Thresh_Bınary)(60-255 BLACCK THRESH BINATY_IN)
plt.figure()
plt.imshow(thresh_img, cmap = "gray")
plt.show()

# Adaptive Thresholding(I think best)
thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,8)
plt.figure()
plt.imshow(thresh_img2, cmap = "gray")
plt.axis("off")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()