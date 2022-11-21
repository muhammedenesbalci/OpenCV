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

