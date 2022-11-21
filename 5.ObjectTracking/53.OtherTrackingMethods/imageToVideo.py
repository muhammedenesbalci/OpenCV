import cv2
import os

path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\5.ObjectDetection\\53.OtherTrackingMethods\\"
path_images = path + "img1\\"
video_path = path + "createdVideo.mp4"

# List of the images
listOfImages = os.listdir(path_images)

# opening a img file
"""
img = cv2.imread(path_images + listOfImages[3])
cv2.imshow("imageToVideo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Create video writer
fps = 30
width = 1920
height = 1080
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))  # (NAME, OS, speed, (width, height))

for i in listOfImages:
    print(i)
    imgPath = path_images + i
    img = cv2.imread(imgPath)
    video_writer.write(img)

video_writer.release()









