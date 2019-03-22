import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

CI = os.listdir("camera_cal/")
CI.sort()
print(CI)
nx = 9
ny = 6

img = cv2.imread("camera_cal/"+CI[1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corner = cv2.findChessboardCorners(gray, (nx, ny), None)

cv2.drawChessboardCorners(img, (nx, ny), corner, ret)

plt.imshow(img)
plt.show()