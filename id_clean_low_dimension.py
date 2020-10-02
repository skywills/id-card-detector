import glob
import os
import cv2
import numpy as np
from PIL import Image
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", type = str, default="/tmp", help="source folder")

args = vars(ap.parse_args())
types = ('*.jpeg', '*.jpg','*.png','*.gif') # the tuple of file types

def getImageFiles(src):
    files_grabbed = []
    i = 1
    for files in types:
        files_grabbed.extend(glob.glob("{}/{}".format(src, files)))  
    return files_grabbed   

def rotate_image(src):
    #image = Image.open(src)
    #out = image.rotate(90, expand=True)
    #out.save(src)
    print('processing rotate ' + src)
    image = Image.open(src)
    out = cv2.rotate(np.array(image), cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(src,cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

def rotate_images(folder):
    files_grabbed = getImageFiles(folder)
    for file in list(files_grabbed):
        try:
            image = Image.open(file)
            width, height = image.size
        
            if (height > width):
                rotate_image(file)
                
        except:
            print('fail to process image ' + file)


def clean_low_dimension(folder):
    i = 1
    files_grabbed = getImageFiles(folder)
   
    for file in list(files_grabbed):
        try:
            image = Image.open(file)
            width, height = image.size
        
            if not(width>=1500 and height>=1000):
                i+=1
                print("file {}, width: {}, height: {}".format(file, width, height))
                os.remove(file)
        except:
            print('fail to process image ' + file)
        
    print("total {} files with low dimension deleted".format(i))

def main():
    source = args['src']
    rotate_images(source)
    clean_low_dimension(source)


if __name__ == "__main__":
    main()