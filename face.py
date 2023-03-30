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
'''def prepare_training_data(data_folder_path):
 
#------STEP-1--------
#get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
 
#list to hold all subject faces
    faces = []
#list to hold labels for all subjects
    labels = []
 
#let's go through each directory and read images within it
    for dir_name in dirs:
 
#our subject directories start with letter 's' so
#ignore any non-relevant directories if any
 #       if not dir_name.startswith("s"):
  #          continue;
 
#------STEP-2--------
#extract label number of subject from dir_name
#format of dir name = slabel
#, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
 
#build path of directory containing images for current subject subject
#sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
 
#get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
 
#------STEP-3--------
#go through each image name, read image, 
#detect face and add face to list of faces
        for image_name in subject_images_names:
 
#ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
 
#build image path
#sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

#read image
            image = cv2.imread(image_path)
 
#display an image window to show the image 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
 
#detect face
face, rect = detect_face(image)
 
#------STEP-4--------
#for the purpose of this tutorial
#we will ignore faces that are not detected
if face is not None:
#add face to list of faces
    faces.append(face)
#add label for this face
labels.append(label)
 
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
 
return faces, labels
'''
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
