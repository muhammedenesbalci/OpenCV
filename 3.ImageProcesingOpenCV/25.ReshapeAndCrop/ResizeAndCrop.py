import cv2

# Opening Image Normal format
path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\3.ImageProcesingOpenCV\\25.ReshapeAndCrop\\lenna.png"
img_normal = cv2.imread(path)
cv2.imshow("Normal Image", img_normal)

# Image Shape-Normal
print("Normal : ", img_normal.shape) # 512, 512, 3 (3 means RGB)

# Opening Image Gray scale
img_gray = cv2.imread(path, 0)
cv2.imshow("Gray Image", img_gray)

# Image Shape-Gray
print("Gray : ", img_gray.shape)

# Reshape Image
img_reshaped = cv2.resize(img_gray, (800, 800))
cv2.imshow("Resized Image", img_reshaped)

# Crop Image
print("array format : ", img_gray)
img_croped = img_normal[0:400, 0:200]  # think as an array because it is an array
cv2.imshow("Cropped Image", img_croped)

cv2.waitKey(0)
cv2.destroyAllWindows()
