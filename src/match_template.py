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

img_path = sys.argv[1]
template_path = sys.argv[2]


img_rgb = cv2.imread(img_path)
img_rgb = cv2.resize(img_rgb, (1024,1024))
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(template_path, 0)
h, w = template.shape[::]
#methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
# For TM_SQDIFF, Good match yields minimum value; bad match yields large values
# For all others it is exactly opposite, max value = good fit.
plt.imshow(res, cmap='gray')
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)  #White rectangle with thickness 2.
cv2.imshow("Matched image", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()