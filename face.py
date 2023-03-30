import cv2
from PIL import ImageGrab
import numpy as np
import os
# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
# capture frames from a camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
i=0

  
# loop runs if capturing has been initialized.
while True: 
  
    # reads frames from a camera
    ret, img = cap.read() 
    if ret:
    # convert to gray scale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)
    #this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.circle(img,(x,y),radius=0,color=(0,0,255),thickness=10)
        cv2.circle(img,(x,y),radius=0,color=(0,0,255),thickness=10)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        t_img=img.copy()
        t_img=t_img[y-10:y+h+20,x-20:x+w+20]
        #t_img.dtype("ui8")
        status = cv2.imwrite(str(i)+'.jpg',t_img)
        i+=1
  
    # Display an image in a window
    cv2.imshow('img',img)
  
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
# Close the window
cap.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows()
