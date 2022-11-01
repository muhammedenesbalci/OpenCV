# Open image
import cv2

# Normal Format
img_normal=cv2.imread("C:\\Users\\CASPER\\Desktop\\OpenCV\\3.ImageProcesingOpenCV\\messi5.jpg")

# Show Image
cv2.imshow("First Image", img_normal)

# Sometimes this code needed for openin an image, for example pycharm
cv2.waitKey(0)
cv2.destroyAllWindows()

# Connecting keyboard (27 == esc)(s ==s)
key = cv2.waitKey(0) & 0xFF
if key == 27:
    cv2.destroyAllWindows()
elif key == ord('q'):
    cv2.destroyAllWindows()

# Gray Scale
image_gray = cv2.imread("C:\\Users\\CASPER\\Desktop\\OpenCV\\3.ImageProcesingOpenCV\\messi5.jpg", 0)

cv2.imshow("Gray Image", image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving Image
cv2.imwrite("gray_image.png", image_gray)



