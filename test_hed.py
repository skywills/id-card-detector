from utils import img_util
from utils import hed_util
import numpy as np
import cv2
import os

MODEL_NAME = 'model'
HED_NAME = 'HED'
# Grab path to current working directory
CWD_PATH = os.getcwd()
IMAGE_NAME = 'test_images/image1.png'
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
HED_PROTOTEXT_PATH = os.path.join(CWD_PATH,MODEL_NAME,HED_NAME,'deploy.prototxt')
HED_CAFFEE_MODEL_PATH = os.path.join(CWD_PATH,MODEL_NAME,HED_NAME,'hed_pretrained_bsds.caffemodel')

img = cv2.imread(PATH_TO_IMAGE, cv2.COLOR_BGR2RGB)
net = hed_util.load_dnn(HED_PROTOTEXT_PATH,HED_CAFFEE_MODEL_PATH)
hed_edge = hed_util.edge_detection(net, img)
gray, edges = img_util.edge_detection(hed_edge,1,3,0.1, 0.29)
cropped = img_util.border_detection(img, edges)
cv2.imshow('HED DETECT',hed_edge)
cv2.imshow('Canny',gray)
cv2.imshow('cropped',cropped)
cv2.waitKey(0)
