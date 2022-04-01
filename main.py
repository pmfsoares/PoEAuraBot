import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
import sys

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
wincap = WindowCapture('Path of Exile')

# load the trained model
cascade_limestone = cv.CascadeClassifier('cascade/cascade.xml')
# load an empty Vision class
vision_limestone = Vision(None)

loop_time = time()
scPos = False
scNeg = False
sc_Time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    # do object detection
    #rectangles = cascade_limestone.detectMultiScale(screenshot)

    # draw the detection results onto the original image
    #detection_image = vision_limestone.draw_rectangles(screenshot, rectangles)



    # display the images
    cv.imshow('Matches', screenshot)

    # debug the loop r1
    key = cv.waitKey(1)
    now = time()
    

    #Take a screenshot every X seconds
    if(now - sc_Time >= 5):
        sc_Time = now
        if(scPos):
            print("Wrote Positive: positive/{}.jpg".format(loop_time))
            cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
        if(scNeg):
            print("Wrote Negative: negative/{}.jpg".format(loop_time))
            cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)
    
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        print("Started positive capture")
        scPos = not scPos
    elif key == ord('d'):
        print("Started Negative capture")
        scNeg = not scNeg
print('Done.')
