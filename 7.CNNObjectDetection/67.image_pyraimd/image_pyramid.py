# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import os

def image_pyramid(image, scale_ratio = 1.5, minSize = (10,10)):
    
    images = []
    while True:
        widthAndheight = int(image.shape[1]/scale_ratio)
        image = cv2.resize(image, dsize = (widthAndheight, widthAndheight))
        images.append(image)
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:  # yazdırıekn ters yazıyorud unutma o yüzden bu şekilde alıyourz  y x şeklinde evriyoe yani height ve width
            break
        
    return images

img = cv2.imread("C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\7.CNNObjectDetection\\67.image_pyraimd\\husky.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

a = 0
for i in image_pyramid(img):
    plt.figure, plt.imshow(i),plt.title(a), plt.show()
    a += 1