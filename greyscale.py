import os
import cv2
import numpy as np
import sys
import glob
import time
from shutil import copyfile

from PIL import Image

MODEL_NAME = 'model'
HED_NAME = 'HED'
# Grab path to current working directory
CWD_PATH = os.getcwd()


PATH_TO_GRAY = ''

def process_image(image_path):
    image = cv2.imread(image_path)
    #gray, edges = edge_detection(img,resize_scale=1, sigma=0.5, l_thresh=0.1, h_thresh=0.29)
    #image = border_detection(gray, edges)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_path, gray)



lst = sorted(glob.glob("{}/*".format(PATH_TO_GRAY)))
i = 1
start = time.time()
print('beginning process image from folders')
for file in list(lst):
    process_image(file)


