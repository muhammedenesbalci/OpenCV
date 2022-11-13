"""
keyboard module keyboard actions
uuid random id
mss screenshot module

https://fivesjs.skipser.com/trex-game/
"""

# Librarys
import keyboard
import time
from PIL import Image
from mss import mss


# Screenshot the screen
def recordScreen(pressedKey, coordinates, path):
    global pressedKeyCounter
    pressedKeyCounter +=1

    print("{} : {}".format(pressedKey, pressedKeyCounter))

    screenRecorderModule = mss()
    img = screenRecorderModule.grab(coordinates)

    lastImg = Image.frombytes("RGB", img.size, img.rgb)
    lastImg.save(path + "{}_{}.jpg".format(pressedKey, pressedKeyCounter))



pressedKeyCounter = 0
cordinatesDict = {"top" : 383, "left" : 721, "width" : 171, "height" : 121}
imagesPath = "C:\\Users\\CASPER\\Desktop\\OpenCV\\MyCodes\\6.ConvolutionalNeuralNetworks\\DataCollecting\\images\\"

while True:

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            recordScreen("up", cordinatesDict, imagesPath)
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            recordScreen("down", cordinatesDict, imagesPath)
            time.sleep(0.1)
        elif keyboard.is_pressed("left"):
            recordScreen("left", cordinatesDict, imagesPath)
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            recordScreen("right", cordinatesDict, imagesPath)
            time.sleep(0.1)

    except :
        continue