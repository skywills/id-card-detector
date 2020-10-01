
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob

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

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def crop_id_image(src): 
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(src)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.60)

    ymin, xmin, ymax, xmax = array_coord

    shape = np.shape(image)
    im_width, im_height = shape[1], shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    # Using Image to crop and save the extracted copied image
    im = Image.open(src)
    return im.crop((left, top, right, bottom))

def id_detect(image, normalized = False): 
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value    
    ori = image.copy()
    image_expanded = np.expand_dims(image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=normalized,
        line_thickness=3,
        min_score_thresh=0.60)

    ymin, xmin, ymax, xmax = array_coord

    shape = np.shape(image)
    im_width, im_height = shape[1], shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    # Using Image to crop and save the extracted copied image
    im = Image.fromarray(ori)
    return im.crop((left, top, right, bottom))    

def clean_edge(image):
    gray, edges = img_util.edge_detection(np.array(image),resize_scale=1, sigma=0.5, l_thresh=0.1, h_thresh=0.29)
    gray = img_util.border_detection(gray,edges)
    return cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

# crop image method, can be edge detection, then crop
def crop_image(src_path, targe_path):
    #img = Image.open(src_path)
    img = cv2.imread(src_path, cv2.COLOR_BGR2RGB)
    im_edge = clean_edge(img)
    im = id_detect(im_edge)
    im.save(targe_path, quality=95)

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
    for file in list(lst):
        output_path = "{}/{}".format(target,os.path.basename(file))
        try:
            crop_image(file, output_path)
        except:
            print("failed on process {}".format(file))
        print(" processing {} image".format(i))
        i+=1

def test_crop():
    test_image = CWD_PATH + "/test_images/image1.png"
    output_folder = CWD_PATH + "/output"
    crop_image(test_image,output_folder + "/cropped.jpg")
    crop_image(test_image,output_folder + "/cropped2.jpg")

def main():
    #test_crop()
    crop_from_folder()




if __name__ == "__main__":
    main()