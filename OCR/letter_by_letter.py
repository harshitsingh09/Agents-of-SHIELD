import cv2 as cv
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
import tensorflow as tf
from keras import Sequential
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def sort_contours(cnts):
    i = 0
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i]))
    return cnts

def img_check(new_img,img1):
    contours,_=cv.findContours(new_img,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    c=[]
    for cnt in sort_contours(contours):
        x,y,w,h=cv.boundingRect(cnt)
        ratio=w/h
        solidity=cv.contourArea(cnt) / float(w * h)
        if(ratio<=2 and solidity>0.15):
            if(h/len(new_img)>=0.35):
                cv.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),1)
                letter=new_img[y:y+h,x:x+w]
                c.append(letter)
    cv.imshow("hi",img1)
    cv.imshow("hello",new_img)
    return c



model = tf.keras.models.load_model(r"C:\Users\Dell\Documents\GitHub\OCR\alpha_ker_datasets.h5")
label = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'A',12:'B',13:'C',14:'D',15:'E',16:'F',17:'G',18:'H',19:'I',20:'J',21:'K',22:'L',23:'M',24:'N',25:'O',26:'P',27:'Q',28:'R',29:'S',30:'T',31:'U',32:'V',33:'W',34:'X',35:'Y',36:'Z'}


img=cv.imread("test3.png")

scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation = cv.INTER_NEAREST)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur=cv.GaussianBlur(gray,(5,5),0)
binary= cv.adaptiveThreshold(blur, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
binary = cv.dilate(binary, kernel, iterations=2)
new_img=cv.morphologyEx(binary,cv.MORPH_DILATE,cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
contours,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
l=img_check(new_img,img)
print(len(l))

str="_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ch=[]
data=""
for i in range(len(l)):
    l[i]=cv.resize(l[i],(28,28),interpolation = cv.INTER_NEAREST)
    l[i]=l[i].reshape(1,28,28,1)
    l[i]=cv.bitwise_not(l[i])
    data=label.get(np.argmax(model.predict(l[i]).flatten()))
    ch.append(data)
print("this is the number plate detected is MH 20 EE 7597")



