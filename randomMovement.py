# call this to spazz out your computer

import pyautogui as gui
import time
import random

k = random.randint(0, 4)
print(k)
time.sleep(2)
keys = ["w", "a", "s", "d", "space", "shift"]
spec = ["space", "space", ""]
starttime = time.time()

while(time.time()-starttime < 60):
    k = random.randint(0, 5)
    print(k)
    gui.keyDown(keys[k])
    gui.keyDown(spec[s])
    time.sleep(0.15)
    gui.keyUp(keys[k])
    gui.keyUp(spec[s])
