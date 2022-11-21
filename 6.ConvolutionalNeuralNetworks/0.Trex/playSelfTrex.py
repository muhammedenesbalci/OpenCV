from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

# Screen Shot
mon = {"top" : 383, "left" : 721, "width" : 171, "height" : 121}
sct = mss()

width = 125
height = 50

# Load Model
model = model_from_json(open("model_new.json", "r").read()) # load model
model.load_weights("trex_weight_new.h5")  # load weight

# down = 0, up = 2
labels = ["Down", "Up"]

framerate_time = time.time()
counter = 0
i = 0
delay = 0.4
key_down_pressed = False
while True:

    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255

    X = np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
    r = model.predict(X)
    print(X.shape[0])

    print("resultsAll : ", r)

    result = np.argmax(r) #  arrayin içdneki en büyükj değer sahip olanın indexini döndüürüyor
    print("resultMax : ", result)

    if result == 0:  # down = 0

        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True

    elif result == 1:  # up = 2

        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)

        if i < 1500:
            time.sleep(0.3)
        elif 1500 < i and i < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)

        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)

    counter += 1

    if (time.time() - framerate_time) > 1:

        counter = 0
        framerate_time = time.time()
        if i <= 1500:
            delay -= 0.003
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0

        print("---------------------")
        print("Down: {} \nRight:{} \n".format(r[0][0], r[0][1]))
        i += 1



