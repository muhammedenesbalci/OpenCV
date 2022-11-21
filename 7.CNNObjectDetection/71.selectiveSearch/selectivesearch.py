# -*- coding: utf-8 -*-
import cv2
import random
import matplotlib.pyplot as plt

image = cv2.imread("husky.jpg")
image = cv2.resize(image, dsize = (600,600))
plt.figure(), plt.imshow(image), plt.show()


# ilklendir ss
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("start")
rects = ss.process()

output = image.copy()

for (x,y,w,h) in rects[:50]:
    color = [random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output, (x,y),(x+w,y+h),color,2)
    
plt.figure(), plt.imshow(output), plt.show()

