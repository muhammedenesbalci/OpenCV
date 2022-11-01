import cv2
import numpy as np
from _collections import deque

"""
# Take a picture

cap=cv2.VideoCapture(1)

width = cap.get(3)
height = cap.get(4)


path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\4.ObjectDetection\\41.Colordetection\\img.jpg"

while True:
    success, frame = cap.read()

    if success:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(path, frame)
            print("Saved !")

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()

"""

"""
RGB-HSW
RED, GREEN, BLUE --- HUE, SATURATION, VALUE
HUE --> Represent Colors
Saturation --> Turkish mean is 'Doygunluk'
Value --> Brightness 
"""

# Determine deque
sizeOfDeque = 16
dequeList = deque(maxlen=sizeOfDeque)

# HSV values
"""
- Browse hsv ranges of individual points of the object
HSV --> (184, 64, 67)
Lower --> (84, 98, 0)
Upper --> (184, 255, 255)
"""

# Capture
cap = cv2.VideoCapture(1)
cap.set(3, 960)  # width
cap.set(4, 480)  # height

# HSV Bounds in lessons
lower_hsv = (84, 98, 80)
higher_hsv = (184, 255, 255)

while True:

    success, frame = cap.read()

    if success:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("Video", frame_hsv)

        # masking with hsv
        masked = cv2.inRange(frame_hsv, lower_hsv, higher_hsv)
        # cv2.imshow("Video", masked)

        # Morphology operations
        eroding_img = cv2.erode(masked, None, iterations=2)
        # cv2.imshow("Video", eroding_img)

        dilation_img = cv2.dilate(eroding_img, None, iterations=2)
        cv2.imshow("Video", eroding_img)

        (contours, _) = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            """
            print(len(contours))
            print(contours)
            print(type(contours))
            """

            # take the bigger are within contours
            maxContour = max(contours, key=cv2.contourArea)

            # Get the rect
            rect = cv2.minAreaRect(maxContour)
            ((x, y), (width, height), rotation) = rect # x y is the centers points

            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)

            # Draw rect
            """
            cv2.rectangle(frame, (int(x- width/2), int(y- height/2)), (x + int(width/2), y + int(height+2)), (255, 0, 0), 3)
            """
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 3)

            # Draw center
            M = cv2.moments(maxContour)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
