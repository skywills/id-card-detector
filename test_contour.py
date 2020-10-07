from utils import img_util
from utils import hed_util
import numpy as np
import cv2
import os

MODEL_NAME = 'model'
HED_NAME = 'HED'
# Grab path to current working directory
CWD_PATH = os.getcwd()
IMAGE_NAME = 'test_images/001461.jpeg'
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
HED_PROTOTEXT_PATH = os.path.join(CWD_PATH,MODEL_NAME,HED_NAME,'deploy.prototxt')
HED_CAFFEE_MODEL_PATH = os.path.join(CWD_PATH,MODEL_NAME,HED_NAME,'hed_pretrained_bsds.caffemodel')

def hed_edge(image):
    net = hed_util.load_dnn(HED_PROTOTEXT_PATH,HED_CAFFEE_MODEL_PATH)
    hed_edge = hed_util.edge_detection(net, image)
    return cv2.cvtColor(hed_edge,cv2.COLOR_BGR2GRAY)

def canny_edge(image):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)
    edges = cv2.Canny(gray,400,600,apertureSize = 5)
    return edges  

def auto_canny(image, sigma=0.33):
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged   

img = cv2.imread(PATH_TO_IMAGE, cv2.COLOR_BGR2RGB)
gray = hed_edge(img)
binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnt = None
max_area = x = y = w = h = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > max_area:
        x, y, w, h = cv2.boundingRect(c)
        max_area = area
        cnt = c

cv2.drawContours(img, [cnt], 0, (0,  255, 0), 3)
cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 3)
cv2.imshow('gray ',gray)
cv2.imshow('binary ', binary)
#cv2.imshow(kWinName,image)
cv2.imshow('orgi ', img)
cv2.waitKey(0)
