import cv2
import time

# Video Path
video_path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\3.ImageProcesingOpenCV\\23.OpeningVideo\\MOT17-04-DPM.mp4"

# Transfer video to py file
cap = cv2.VideoCapture(video_path)

# Control can video open
if cap.isOpened() == False:
    print("Error, can not open")

# Height and Weight
print("Height : ", cap.get(3))
print("Width : ", cap.get(4))

# Opening video
while True:
    succes, frame = cap.read()  # succes : can open, frame every frame

    if succes == True:
        time.sleep(0.01)  # If we do not write this, video play more quickly
        cv2.imshow("Video", frame)
    else:
        print("Error")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()  # stop capture
cv2.destroyAllWindows()  # close windows
