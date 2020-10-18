
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
from utils.dlib_face_util import DlibHumanFaceDetect
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
PATH_EYE_XML = os.path.join(CWD_PATH, 'model/haarcascade_eye.xml')
PATH_NOSE_XML = os.path.join(CWD_PATH, 'model/haarcascade_mcs_nose.xml')

net = hed_util.load_dnn(HED_PROTOTEXT_PATH,HED_CAFFEE_MODEL_PATH)
face_detect = HumanFaceDetect()
dlib_face_detect = DlibHumanFaceDetect()
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

def save_image(path, img):
    if(is_jpg(path)):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB),[cv2.IMWRITE_JPEG_QUALITY, 85])
    else:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# crop image method, can be edge detection, then crop
def crop_image(src_path, target_path):
    target_path = preprocess_filename(target_path)
    origin = np.array(Image.open(src_path))
    start = time.time()
    hasFace, img = try_rotate_image_with_face_detect(origin)
    print('has face ', hasFace)
    if (hasFace):
        cropped = hed_util.crop_image(net, img)
        if(cropped is None or not face_detect.hasFaces(cropped) or cropped.shape[0] < min_height):
            cropped = img
            if(cropped is None):
                print('failed to crop image')
            elif (cropped.shape[0] < min_height):
                print('does not meet min height {} img height {}'.format( min_height, cropped.shape[0]))
            else:
                print('no face detected')
            #print('cropped image dont have face or has no contour or does not meet min height {} img height {}'
            #.format( min_height, cropped.shape[0]))
            target_path = rename_withprefix(target_path,'origin')
        else:
            #cropped = try_rotate_image(cropped)
            target_path = rename_withprefix(target_path,'cropped')
    else:
        cropped = origin
        print('failed to detect face while try rotate image')
    print("image: {}, used time: {}".format(src_path, time.time() - start))
    save_image(target_path, cropped)


def try_rotate_image_with_face_detect(img):
    # rotate with 4 angle, 0, 90, 180,270
    for i in range(3):
        print('trying to rotate image counterclockwise to {} degree'.format(i * 90))
        raw = img if i == 0 else np.rot90(img,i)
        hasFace = dlib_face_detect.hasFaces(raw)
        if(hasFace):
            return True,raw
    return False,img

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
    test_crop()
    #crop_from_folder()




if __name__ == "__main__":
    main()