import tensorflow as tf
from tensorflow import keras
import numpy as np
model = keras.models.load_model("cs_aimv2.h5")

#model.predict()

from PIL import Image, ImageGrab
import pyautogui as gui
import time

winx = 1920
winy = 1200

midx = winx/2
midy = winy/2

rad = 74/2

box = (midx - rad, midy - rad, midx + rad, midy + rad)

time.sleep(1)

starttime = time.time()

while(True):
    im = ImageGrab.grab(bbox = box)

    rgb_im = im.convert('RGB')

    print("started")
    print(np.array(rgb_im).shape)

    if(model.predict(np.array([np.array(rgb_im)])) > np.array([[0.5]])):
       
        gui.move(0,-80)
    else:
        print("nothing")
    