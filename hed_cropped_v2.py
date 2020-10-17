
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
import time
from shutil import copyfile
from utils.face_util import HumanFaceDetect
from PIL import Image

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", type = str, default="", help="source folder")
ap.add_argument("-t", "--target", type = str, default="", help="target folder")

args = vars(ap.parse_args())

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import img_util
from utils import hed_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'
HED_NAME = 'HED'
# Grab path to current working directory
CWD_PATH = os.getcwd()
HED_PROTOTEXT_PATH = os.path.join(CWD_PATH,MODEL_NAME,HED_NAME,'deploy.prototxt')
HED_CAFFEE_MODEL_PATH = os.path.join(CWD_PATH,MODEL_NAME,HED_NAME,'hed_pretrained_bsds.caffemodel')
PATH_FACE_XML = os.path.join(CWD_PATH, 'model/haarcascade_frontalface_alt.xml')

net = hed_util.load_dnn(HED_PROTOTEXT_PATH,HED_CAFFEE_MODEL_PATH)
face_detect = HumanFaceDetect(PATH_FACE_XML)
min_height = 1000

def rename_withprefix(path, prefix):
   origin_filename =  os.path.basename(path)
   new_filename = path.replace(origin_filename,'{}_{}'.format(prefix,origin_filename.lower()))
   return new_filename

def preprocess_filename(path):
    return path.replace('jpeg','jpg')

def is_jpg(path):
    extension = os.path.splitext(path)[1]
    return extension == '.jpeg' or extension == '.jpg'

# crop image method, can be edge detection, then crop
def crop_image(src_path, target_path):
    #img = cv2.imread(src_path, cv2.COLOR_BGR2RGB)
    target_path = preprocess_filename(target_path)
    img = np.array(Image.open(src_path))
    start = time.time()
    cropped = hed_util.crop_image(net, img)
    if(cropped is None or not face_detect.hasFaces(cropped) or cropped.shape[0] < min_height):
        cropped = img
        print('cropped image dont have face or has no contour or does not meet min height {} img height {}'
        .format( min_height, cropped.shape[0]))
        target_path = rename_withprefix(target_path,'origin')
    else:
        cropped = try_rotate_image(cropped)
        target_path = rename_withprefix(target_path,'cropped')
    #hed_edge = hed_util.edge_detection(net, im)
    #cropped = clean_edge(im, hed_edge)
    print("image: {}, used time: {}".format(src_path, time.time() - start))
    if(is_jpg(target_path)):
        cv2.imwrite(target_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),[cv2.IMWRITE_JPEG_QUALITY, 85])
        #cv2.imwrite(target_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(target_path, cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

def try_rotate_image(img):
    height, width, _ = img.shape
    if(height > width):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def crop_from_folder():
    source = args['src']
    target = args['target']
    print("src: ", source)
    print("target: ", target)
    if not source:
        source = CWD_PATH
    if not target:
        target = CWD_PATH  
    lst = sorted(glob.glob("{}/*".format(source)))
    i = 1
    start = time.time()
    print('beginning process image from folders')
    for file in list(lst):
        output_path = "{}/{}".format(target,os.path.basename(file))
        try:
            crop_image(file, output_path)
        except:
            print("failed on process {}, copy image to dest folder".format(file))
            copyfile(file, output_path)
        print(" processing {} image".format(i))
        i+=1
    print("processed {} images, total used time: {}".format(i, time.time() - start))

def test_crop():
    test_image = CWD_PATH + "/test_images/001461.jpeg"
    output_folder = CWD_PATH + "/output"
    crop_image(test_image,output_folder + "/Cropped.jpeg")
    crop_image(test_image,output_folder + "/cropped2.jpg")

def main():
    #test_crop()
    crop_from_folder()




if __name__ == "__main__":
    main()