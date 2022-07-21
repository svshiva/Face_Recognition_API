import cv2
import numpy as np
from urllib.request import urlopen
import face_recognition


def get_image_from_url(url):
    img = urlopen(url)
    img = np.array(bytearray(img.read()), dtype=np.uint8)
    img = cv2.imdecode(img, -1)
    return img

def get_image_from_path(image_path):
    image=cv2.imread(image_path)
    return image
    
def predict(image,test_image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    test_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    face_encodings=face_recognition.face_encodings(image)[0]
    face_encodings_test=face_recognition.face_encodings(test_image)[0]
    result=face_recognition.compare_faces([face_encodings],face_encodings_test)
    distance = face_recognition.face_distance([face_encodings],face_encodings_test)
    return result,distance
