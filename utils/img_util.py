
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

def findContour(gray, convert_binary=False):
    if(convert_binary):
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

def try_crop_image(image, base_width=860, threshold=0.3, iterations=1):
    print('with base width, ', base_width)
    cropped = crop_image(image)
    # 1. trying to crop with simple threshold border dection
    if(is_cropped(image, cropped) and not is_too_small(image, cropped, threshold=threshold)):
        return cropped
    
    # 2. retry using dilate inner content with dilate iterations
    cropped = dilate_crop_image(image,  iterations=iterations)

    if(is_cropped(image, cropped) and not is_too_small(image, cropped, threshold=threshold)):
        return cropped

    cropped = dilate_crop_image_with_scale(image, base_width, iterations=iterations)
    if(is_cropped(image, cropped) and not is_too_small(image, cropped, threshold=threshold)):
        return cropped    

    return image    

def is_cropped(src, target):
    height, width = src.shape[:2]
    target_height, target_width = target.shape[:2]
    return target_height < height or target_width < width

def is_too_small(src, target, threshold=0.3):
    height, width = src.shape[:2]
    target_height, target_width = target.shape[:2]
    min_ratio = 1 - threshold
    min_width = width * min_ratio
    min_height = height * min_ratio
    return  target_width < min_width or target_height < min_height 


def crop_image(image):
    if(len(image.shape)==2):
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 90, 90)
    contours, last_cnt = findContour(gray, convert_binary=True)
    if (last_cnt is None):
        return None
    x,y,w,h = cv2.boundingRect(contours[len(contours)-1])
    #print('x {} y {} w {} h {}'.format(x,y,w,h))
    return image[y:y+h, x:x+w]

def dilate_crop_image(image, iterations=3):
    (x,y,w,h) = dilate_inner_get_contour(image, iterations=iterations)
    return image[y:y+h, x:x+w]

def dilate_crop_image_with_scale(image, base_width, iterations=3):
    scale = 1
    base_kernel = 29
    if (image.shape[1] > base_width):
        scale = image.shape[1] // base_width
        scale = 2 if scale == 1 else scale
    kernel_size = base_kernel * scale
    (x,y,w,h) = dilate_inner_get_contour(image, iterations=iterations, kernelSize=kernel_size)
    return image[y:y+h, x:x+w]    

def dilate_inner_get_contour(src, iterations = 1, kernelSize = 17):
    if(len(src.shape)==2):
        gray = src
    else:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #blur image
    blur = cv2.medianBlur(gray,7)
    #cleaning noise
    thresh1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,11, 9)  

    # Specify structure shape and kernel size.  
    # Kernel size increases or decreases the area  
    # of the rectangle to be detected. 
    # A smaller value like (10, 10) will detect  
    # each word instead of a sentence. 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize)) 
    
    # Appplying dilation on the threshold image 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = iterations)     

    # Finding dilation contours 
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                    cv2.CHAIN_APPROX_SIMPLE)     

    cnt = max(contours, key=cv2.contourArea)
    x1, y1, w1, h1 = cv2.boundingRect(cnt) 
    #textarea = src[y1:y1+h1, x1:x1+w1]
    #return img_util.sharpen(textarea)
    return (x1, y1, w1, h1)

def sharpen(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)        


def denoise(image, scale=1, iterations=3, initialblockSize=23, initialSigma=11):
    blockSize = initialblockSize * scale
    sigma = initialSigma * scale
    thresh1 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,blockSize, sigma)  
    return cv2.fastNlMeansDenoising(thresh1, None, 51,7,31)
    