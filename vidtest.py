# Separate video into images programmatically

import cv2
import numpy as np

cap = cv2.VideoCapture('CSv2.mp4')

count = 0

success = 1
while success: 
    success, image = cap.read() 
    cv2.imwrite("CS_Data/Train/frame%d.jpg" % count, image)
    count += 1