import cv2
from PIL import Image
from skimage import io
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
import cv2
import glob
from matplotlib import pyplot as plt

import sys

image_path = sys.argv[1]
crop_out_path = sys.argv[2]
cropping = True

x_start, y_start, x_end, y_end = 0, 0, 0, 0
image = cv2.imread(image_path)
image = cv2.resize(image, (1024,1024))
oriImage = image.copy()
def mouse_crop(event, x, y, flags, param):
    
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
#             print(roi.shape)
#             h= roi.shape[0] 
#             w = roi.shape[1]
            cv2.imshow("Cropped", roi)
            #print(x, y)   
            cv2.imwrite(crop_out_path,roi)
            cv2.waitKey(0)
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
# while True:
#     i = image.copy()
#     if not cropping:
#         cv2.imshow("image", image)
#     elif cropping:
#         cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 255, 0), 3)
#     cv2.imshow("image", i)
#     cv2.waitKey(0)
#     # close all open windows
#     cv2.destroyAllWindows()