import os
import dlib
import cv2

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
PREDICTOR_PATH = os.path.join(PARENT_PATH, 'model/shape_predictor_5_face_landmarks.dat')
'''
Implement 5 point detection for human faces
'''
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y      
    return (x, y, w, h)

def cropFace(face, img):
    face_x, face_y, face_w, face_h = rect_to_bb(face)
    return img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]

class DlibHumanFaceDetect:

    def __init__(self, predictor_path=PREDICTOR_PATH):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)


    def detectFaces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector(gray, 0)

    def detectFacesImages(self, img):
        faces = self.detectFaces(img)
        imgs = []
        for face in faces:
            img = cropFace(face, img)
            imgs.append(img)

        return imgs

    def detectFirstFace(self, img):
        faces = self.detectFaces(img)  
        if(len(faces) > 0):
           return cropFace(faces[0], img)
        else:
            return None

    def hasFaces(self, img):
        # scaleFactor=1.2, minNeighbors=3, minSize=(48, 48)
        faces = self.detectFaces(img)
        return len(faces) > 0
