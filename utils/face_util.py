import cv2

"""
def init_classifiers(face_path, eye_path):
    face_classifier = cv2.CascadeClassifier(face_path)
    eye_classifier = cv2.CascadeClassifier(eye_path)    
    return face_classifier, eye_classifier


def init_face_classifier(face_path):
    return cv2.CascadeClassifier(face_path)
"""

class HumanFaceDetect:

    def __init__(self, frontal_face_xml):
        self.frontal_face_xml = frontal_face_xml
        #self.eye_glasses_xml = eye_glasses_xml    
        self.face_classifier = cv2.CascadeClassifier(self.frontal_face_xml)
        #self.eye_classifier = cv2.CascadeClassifier(self.eye_glasses_xml)


    def detectFaces(self, img, scaleFactor=None, minNeighbors=None, minSize=None):
        # scaleFactor=1.2, minNeighbors=3, minSize=(48, 48)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.face_classifier.detectMultiScale(gray,scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

    def hasFaces(self, img, scaleFactor=None, minNeighbors=None, minSize=None):
        # scaleFactor=1.2, minNeighbors=3, minSize=(48, 48)
        faces = self.detectFaces(img,scaleFactor, minNeighbors, minSize)
        return len(faces) > 0
