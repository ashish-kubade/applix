import cv2
import numpy as np


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
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')
threshold = 0.8
loc = np.where( res >= threshold)  
for pt in zip(*loc[::-1]):  
       cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()