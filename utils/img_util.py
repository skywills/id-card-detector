
import cv2
import numpy as np
import skimage
import skimage.feature
from skimage.color import rgb2gray
from skimage.transform import resize, rescale

def edge_detection(image,resize_scale, sigma, l_thresh, h_thresh):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray,(image.shape[1]//resize_scale,image.shape[0]//resize_scale))
#     rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
    #blur = cv2.GaussianBlur(resized, (5, 5),0)
    blur = cv2.blur(resized,(5,5))
    
    edges = skimage.feature.canny(
    image=blur/255.0,
    sigma=sigma,
    low_threshold=l_thresh,
    high_threshold=h_thresh,
    )
    return image, edges

def border_detection(image,edges):
    x = [i for i in range(edges.shape[0]) if np.count_nonzero(edges[i] == True, axis = 0)>0]
    
#     for i in range(0,edges.shape[0]):
#         if (edges[i].any() == True):
#             x.append(i)
    y = [i for i in range(edges.shape[1]) if np.count_nonzero(edges[:,i] == True, axis = 0)>0]
#     for i in range(0,edges.transpose().shape[0]):
#         if (edges.transpose()[i].any() == True):
#             y.append(i)
    if ((len(x)>0) and (len(y)>0)):
        image = image[min(x):max(x),min(y):max(y)]
    
    return image# , min(x),max(x),min(y),max(y)

def findContour(gray):
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    contours = []
    last_cnt = None
    max_area = x = y = w = h = 0


    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            x, y, w, h = cv2.boundingRect(c)
            max_area = area
            last_cnt = c
            contours.append(c)

    return contours, last_cnt

