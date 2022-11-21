#%% importings
# -*- coding: utf-8 -*-
import cv2
import pickle
import numpy as np
import keras


def preProcess(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255
    return img

"""
pickle_in = open("C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\1.RealTimeNumberClassification\\model_trained.p", "rb")
model = pickle.load(pickle_in)
"""
# %% load model and wights
from keras.models import load_model
from keras.models import Model  # for summary

model = load_model("C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\1.RealTimeNumberClassification\\models\\model\\model.h5")
model.summary()


# %% singel image

img = cv2.imread("C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\1.RealTimeNumberClassification\\singleImage\\1.png")
img = preProcess(img)
img = np.asarray(img)
img  = img.reshape(1, 32, 32, 1)

pred_res = model.predict(img)
pred_val = np.amax(pred_res)
pred_class = np.argmax(pred_res, axis=1)

# %%start
cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(3, 480)

while True:
    succes, frame = cap.read()
    if succes:
        
        img = preProcess(frame)
        img = np.asarray(img)
        print(img.shape)
        img = img.reshape(1, 32, 32, 1)
        
        # predict result
        prediction_res = model.predict(img)
        prediction_res_index_val = np.amax(prediction_res)  # en yüksek indexdeki value yi vericek
        predicted_class = np.argmax(prediction_res, axis=1)
        
        print(predicted_class, prediction_res_index_val)
        
        if prediction_res_index_val >= 0.6:
            cv2.putText(frame, str(predicted_class) + " : " + str(prediction_res_index_val), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)
            
        cv2.imshow("Classification", frame)
        
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        
    else:
        print("Error")
    
cap.release()
cv2.destroyAllWindows()