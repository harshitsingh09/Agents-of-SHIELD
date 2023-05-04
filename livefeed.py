import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import pickle
from  keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

facenet = FaceNet()
face_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = face_embeddings['arr_1']
enocder = LabelEncoder()
enocder.fit(Y)
harcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))
cap = cv.VideoCapture(0)

while cap.isOpened:
    _,frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = harcascade.detectMultiScale(gray_img, 1.3, 5)
    for (x, y, w, h) in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final = enocder.inverse_transform(face_name)[0]
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, str(final), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv.LINE_AA)

    cv.imshow("Face Recog.", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        exit(0)

cap.release()
cv.destroyAllWindows()