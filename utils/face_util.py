import cv2
import os
"""
def init_classifiers(face_path, eye_path):
    face_classifier = cv2.CascadeClassifier(face_path)
    eye_classifier = cv2.CascadeClassifier(eye_path)    
    return face_classifier, eye_classifier


def init_face_classifier(face_path):
    return cv2.CascadeClassifier(face_path)
"""

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
PATH_FACE_XML = os.path.join(PARENT_PATH, 'model/haarcascade_frontalface_alt.xml')
PATH_EYE_XML = os.path.join(PARENT_PATH, 'model/haarcascade_eye.xml')
PATH_NOSE_XML = os.path.join(PARENT_PATH, 'model/haarcascade_mcs_nose.xml')
def cropFace(face, img):
    face_x, face_y, face_w, face_h = face
    return img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]

class HumanFaceDetect:

    def __init__(self, frontal_face_xml=PATH_FACE_XML, eye_glasses_xml=PATH_EYE_XML, nose_xml=PATH_NOSE_XML):
        self.frontal_face_xml = frontal_face_xml
        self.eye_glasses_xml = eye_glasses_xml   
        self.nose_xml =  nose_xml
        self.face_classifier = cv2.CascadeClassifier(self.frontal_face_xml)
        self.eye_classifier = cv2.CascadeClassifier(self.eye_glasses_xml)
        self.nose_classifier = cv2.CascadeClassifier(self.nose_xml)


    def detectFaces(self, img, biggest_only = True, scaleFactor=None, minNeighbors=None, minSize=None):
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE
        # scaleFactor=1.2, minNeighbors=3, minSize=(48, 48)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.face_classifier.detectMultiScale(gray,scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, flags=flags)

    def detectFacesImages(self, img, biggest_only = True, scaleFactor=None, minNeighbors=None, minSize=None):
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE
        # scaleFactor=1.2, minNeighbors=3, minSize=(48, 48)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
        faces = self.face_classifier.detectMultiScale(gray,scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, flags=flags)
        imgs = []
        for face in faces:
            #face_x, face_y, face_w, face_h = face
            img = cropFace(face, img)
            imgs.append(img)

        return imgs

    def detectFirstFace(self, img, biggest_only = True, scaleFactor=None, minNeighbors=None, minSize=None):
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE
        # scaleFactor=1.2, minNeighbors=3, minSize=(48, 48)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
        faces = self.face_classifier.detectMultiScale(gray,scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, flags=flags)      
        if(len(faces) > 0):
           return cropFace(faces[0], img)
        else:
            return None

    def hasFaces(self, img, scaleFactor=None, minNeighbors=None, minSize=None):
        # scaleFactor=1.2, minNeighbors=3, minSize=(48, 48)
        faces = self.detectFaces(img,scaleFactor, minNeighbors, minSize)
        return len(faces) > 0


    def detectEyes(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        return self.eye_classifier.detectMultiScale(gray)

