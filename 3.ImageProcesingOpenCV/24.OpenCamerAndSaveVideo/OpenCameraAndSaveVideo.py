import cv2

# Local cam video capture
cap = cv2.VideoCapture(0)

# Height and width
# First method
width = cap.get(3)
height = cap.get(4)

print(width, height)

# Second Method
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width, height)

# Saving Video
path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\3.ImageProcesingOpenCV\\24.OpenCamerAndSaveVideo\\video_saving.mp4"
# writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"DIVX"), 20, (width, height))  # (NAME, OS, fps, (width, height))
writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width, height))  # (NAME, OS, speed, (width, height))

while True:
    succes, frame = cap.read()
    
    if succes == True:
        cv2.imshow("video", frame)

        # Save
        writer.write(frame)

        # cv2.imwrite("a.png", frame)  # saving a frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Stop capture
writer.release()  # Stop writing
cv2.destroyAllWindows()  # Destroy windows
