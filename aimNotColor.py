# Just grabs pixel values

from PIL import Image, ImageGrab
import pyautogui as gui
import time

winx = 1920
winy = 1200

midx = winx/2
midy = winy/2

rad = 50

box = (midx - rad, midy - rad, midx + rad, midy + rad)

time.sleep(5)

starttime = time.time()

while(True):
    im = ImageGrab.grab(bbox = box)
    width, height = im.size
    rgb_im = im.convert('RGB')
    r, g, b = rgb_im.getpixel((width/2, (height/2) +11))
    #add 11
    if(g < 210):
        print("gotcha bitch")
        gui.move(0,-20)
    else:
        print("nothing")
    
