#%% Libraries
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import pickle
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import keras

#%% Flowing Images

images_directory_path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\1.RealTimeNumberClassification\\myData"
images_directory_list = os.listdir(images_directory_path)
numberOfClasses = len(images_directory_list)

#%% Preprocces

def preProcess(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255
    return img

images_list = []
classes_list = []

for i in range(0, numberOfClasses):
    myImgList = os.listdir(images_directory_path + "\\" + str(i))
    
    for j in myImgList:
        img = cv2.imread(images_directory_path + "\\" + str(i) + "\\" + j)
        img = preProcess(img)
        images_list.append(img)
        classes_list.append(i)

print(len(images_list))
print(len(classes_list))

#%% Convert list to an array

images_list_array = np.array(images_list) # number of images width, height, dimension  
classes_list_array = np.array(classes_list) 

print(images_list_array.shape, classes_list_array.shape)

#%% Split data
X_train, x_test, Y_train, y_test = train_test_split(images_list_array, classes_list_array, test_size = 0.2, random_state = 42)  # testi train ederken hiç bir zaman görmicek en son eğitim bittiğinde kullanıcaz
X_train, x_validation, Y_train, y_validation = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)  # valid seti eğitirken kullanıcaz normal doğrulamak için

print(X_train.shape)
print(x_test.shape)
print(x_validation.shape)
print("------------------")
print(Y_train.shape)
print(y_test.shape)
print(y_validation.shape)

#%% Arrange dims toplam resim sayısı, width, height, dimension
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)  # zaten 1 ama 1 olduğu zaman yazmıyor fakar eğitirken fit bizden bunun eklememiziz istiyor o yüzden ekliyoruz dimension sayısı
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 32, 32, 1)
print(X_train.shape, x_test.shape, x_validation.shape)

# %% Data Generate
dataGen = ImageDataGenerator(width_shift_range = 0.1, 
                             height_shift_range = 0.1, 
                             zoom_range = 0.1, 
                             rotation_range = 10)

dataGen.fit(X_train) # biz direkt aktardık flow ferom dir diyerekde yukarda yaptığmz pre procces işlemlerini yapabilrdik biz onları kendimiz yaptık

#%% labeling bunuda flow from dic diyerek yapabilrdik aslında buna gerek kalmazdı
Y_train = to_categorical(Y_train, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)
y_validation = to_categorical(y_validation, numberOfClasses)
print(Y_train, y_test, y_validation)
# %% Create Model CNN

model = Sequential()
model.add(Conv2D(input_shape = (32, 32, 1), filters= 8, kernel_size = (5, 5), activation = "relu", padding = "same")) # padding 1 sıra piksel ekler
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters= 8, kernel_size = (3, 3), activation = "relu", padding = "same")) # padding 1 sıra piksel ekler
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2)) # blocking overfitting
model.add(Flatten())

# ANN
model.add(Dense(units = 256, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(units = numberOfClasses, activation = "softmax"))

# %% save a file to see
def myprint(s):
    with open('C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\1.RealTimeNumberClassification\\modelSummary.txt', 'a') as f:
        print(s, file=f)

model.summary(print_fn=myprint)

# %% Compile


model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])

batch_size = 250

hist = model.fit(dataGen.flow(X_train, Y_train, batch_size = batch_size), # Modele dataları vermek için kullanıyoruz datagenerate etme özelliklerini kullanmasan bile bu şekilde yapılıyor
                                        validation_data = (x_validation, y_validation),
                                        epochs = 15, shuffle = 1)
# %% save model and wights
model_path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\1.RealTimeNumberClassification\\models\\model\\model.h5"
model.save(model_path)

"""
pickle_out = open("C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\1.RealTimeNumberClassification\\model_trained_v2.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()
"""

# %% Evulation on test set
hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val Accuracy")
plt.legend()
plt.show()


score = model.evaluate(x_test, y_test, verbose = True)  # verbose = 1 görselleştir demek
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

# %%
y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1) # max değere sahip olan işndexi veriyor
Y_true = np.argmax(y_validation, axis = 1) 
cm = confusion_matrix(Y_true, y_pred_class)
f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show()








































