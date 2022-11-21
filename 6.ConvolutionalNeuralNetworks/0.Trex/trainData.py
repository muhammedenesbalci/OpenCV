import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

# Transfer Images
path = "C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\0.Trex\\DataCollecting\\images\\"
imageNameFiles = os.listdir(path)
labelName = []
imagesList = []


# Image operations
width = 125
height = 50

for i in imageNameFiles:
    label_name = i.split("_")[0]
    labelName.append(label_name)

    # resize image
    img = cv2.imread(path + i, 0)
    img = cv2.resize(img, (width,height))

    # Normalize image pixels values
    img = img / 255
    imagesList.append(img)

# convert list to an array 
imagesList = np.array(imagesList)
print(imagesList.shape, imagesList.size)
print(imagesList.shape[0])


# convert it to suitable format imagelist.shape direkt kaç resim olduğunu da yazabilrdik
imagesList = imagesList.reshape(imagesList.shape[0], width, height, 1) # count of channel last index is 1 we transfer images as gray scale(aslında böyle bir index var ama 1(gray scaleden dolayı) olduğu için yazmıyor biz direkt ekliyoruz) eğer eklerken direkt rgb olarak eklseydik bu 3 olurdu
X = imagesList
print(imagesList, "_-_", imagesList.shape, "_-_", imagesList.size)

# labelEncoder for labelnames 0 down 1 up
labelEncoder = LabelEncoder()
labelEncoded = labelEncoder.fit_transform(labelName)
print(labelEncoded, "_-_", labelEncoded.size,"_-_", labelEncoded.shape)

# Label hot encoding
oneHotEncoder = OneHotEncoder(sparse = False)  # We do not want to sparse matrix
labelEncoded = labelEncoded.reshape(len(labelEncoded), 1) # 169,1 olmasını istedi,k boşluk olmasın diye
print(labelEncoded.shape)

labelOneHotEncoded = oneHotEncoder.fit_transform(labelEncoded)
Y = labelOneHotEncoded
print(labelOneHotEncoded, "_-_", labelOneHotEncoded.shape, "_-_", type(labelOneHotEncoded))

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=2)

# CNN Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(2, activation = "softmax"))

# save a file to see
def myprint(s):
    with open('modelSummary.txt', 'a') as f:
        print(s, file=f)

model.summary(print_fn=myprint)


"""
if os.path.exists("./trex_weight.h5"):
    model.load_weights("trex_weight.h5")
    print("Weights yuklendi")
"""


model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

model.fit(train_X, train_Y, epochs=35, batch_size=32)

score_train = model.evaluate(train_X, train_Y)
print("Training accuracy: %", score_train[1] * 100)

score_test = model.evaluate(test_X, test_Y)
print("Test accuracy: %", score_test[1] * 100)

# saving model weight for using later
open("model_new.json", "w").write(model.to_json())
model.save_weights("trex_weight_new.h5")