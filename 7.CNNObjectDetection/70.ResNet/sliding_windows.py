# -*- coding: utf-8 -*-
import cv2 
import matplotlib.pyplot as plt

def sliding_window(image, pixel_step, sizeOfArea):
    
    images_list = []
    
    for y in range(0, image.shape[0] - sizeOfArea[1], pixel_step):
        for x in range(0, image.shape[1] - sizeOfArea[0], pixel_step):
            
            images_list.append([x, y, image[y : y + sizeOfArea[1], x : x + sizeOfArea[0]]])
            
    return images_list

