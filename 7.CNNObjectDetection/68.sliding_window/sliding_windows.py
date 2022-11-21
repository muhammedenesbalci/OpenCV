# -*- coding: utf-8 -*-
import cv2 
import matplotlib.pyplot as plt

def sliding_window(image, pixel_step, sizeOfArea):
    
    images_list = []
    
    for y in range(0, image.shape[0] - sizeOfArea[1], pixel_step):
        for x in range(0, image.shape[1] - sizeOfArea[0], pixel_step):
            
            images_list.append([x, y, image[y : y + sizeOfArea[1], x : x + sizeOfArea[0]]])
            
    return images_list

img = cv2.imread("C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\7.CNNObjectDetection\\67.image_pyraimd\\husky.jpg")
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

a = 0
for i in sliding_window(img, 100, (200, 200)):
    print(i[0], i[1])
    plt.figure, plt.imshow(i[2]),plt.title(a), plt.show()
    a += 1