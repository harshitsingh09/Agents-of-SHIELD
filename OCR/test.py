from tensorflow import keras
import tensorflow as tf
import cv2
from keras import Sequential
import numpy as np

model = tf.keras.models.load_model(r"C:\Users\Dell\Documents\GitHub\OCR\alpha_ker_datasets")
label = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
img=cv2.imread("test3.png")
img = cv2.GaussianBlur(img, (7,7), 0)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(400,440))
_,img=cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img_dilation = cv2.dilate(img, kernel, iterations=2)
img=cv2.morphologyEx(img,cv2.MORPH_DILATE,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
cv2.imshow("test",img)
img=cv2.resize(img,(28,28))
img=img.reshape(1,28,28,1)
model.summary()
print(label.get(np.where(model.predict(img).flatten()==1)[0][0]))
