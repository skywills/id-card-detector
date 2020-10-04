
import cv2
import numpy as np
import skimage
import skimage.feature
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
import skimage.viewer as ImageViewer


def cv2_canny(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)
    edges = cv2.Canny(gray,400,600,apertureSize = 5)
    #cv2.imshow('image',edges)
    #cv2.waitKey(0)
    return edges

def cv2_canny2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)    
    edged = cv2.Canny(gray,30, 200)
    return edged

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


image_file = ''
image = cv2.imread(image_file)
#gray, edges = edge_detection(img,resize_scale=1, sigma=0.5, l_thresh=0.1, h_thresh=0.29)
#image = border_detection(gray, edges)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

cv2.imshow("Original", image)
cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.waitKey(0)
