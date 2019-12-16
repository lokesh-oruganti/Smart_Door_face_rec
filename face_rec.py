import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import urllib.request
import re
from bs4 import BeautifulSoup



def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding




def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    value = 0
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        if name in known_face_names:
            flag = 1
        else:
            flag = 0
        if flag == 1:
            dataX = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=8U2BZKO96F43GB0B&field1=2&field2={}".format(name)).read()
            print(dataX)
            #dataY = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=8U2BZKO96F43GB0B&field2={}".format(face_names)).read()
            #print(dataY)
        else:
            dataX = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=8U2BZKO96F43GB0B&field1=1&field2=Unknown").read()
            print(dataX)
            #dataY = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=8U2BZKO96F43GB0B&field2=3").read()
            #print(dataY)



        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    print(*known_face_names, sep = ", ") 

    # Display the resulting image
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 
           



print(classify_face("test.jpg"))


