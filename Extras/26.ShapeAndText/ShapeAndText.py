import cv2
import numpy as np

img_zeros = np.zeros((512, 512, 3), np.uint8)  # Black Image

print(img_zeros.shape)
cv2.imshow("Img_zeros", img_zeros)

"""
points (Column, Row) (X, Y)
0/0-----X----->
 |
 |
 Y
 |
 |
 v
"""

# Add line (img, (start points), (end points), (B, G, R), thickness )
cv2.line(img_zeros, (100, 100), (412, 412), (0, 0, 255), 5)
cv2.imshow("With line", img_zeros)

# Add Rectangle (img, (start points), (end points), (B, G, R), thickness or filling )
cv2.rectangle(img_zeros, (100, 100), (300, 300), (255, 0, 0), cv2.FILLED)
cv2.rectangle(img_zeros, (200, 200), (312, 312), (0, 255, 0), 5)
cv2.imshow("With Rectangle", img_zeros)

# Add Circle (img, (center points), (B, G, R), thickness or filling )
cv2.circle(img_zeros, (312, 312), 50, (100, 100, 100), 8)
cv2.circle(img_zeros, (312, 312), 30, (10,123,21), cv2.FILLED)
cv2.imshow("With Circle", img_zeros)

# Add Text  (img, Text, (start points), Font, Size, (B, G, R),, thickness )
cv2.putText(img_zeros, "Shapes and Text", (400,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,  255), 1)
cv2.imshow("With Text", img_zeros)


cv2.waitKey(0)
cv2.destroyAllWindows()