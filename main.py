import cv2
import numpy as np
# returns as array numpy
image = cv2.imread('test_image.jpg')
# convert to grayscale
# use just one channel as opposed to three channel for color image
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
# gaussian blue
# reduce noise and smooth image
blur = cv2.GaussianBlur(gray, (5,5), 0)
# display image
cv2.imshow('result', blur)
cv2.waitKey(0)
