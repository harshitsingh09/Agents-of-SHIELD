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
        print(ratio)
        if(ratio<=2 and solidity>0.15):
            if(h/len(new_img)>=0.35):
                cv.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),1)
                letter=new_img[y:y+h,x:x+w]
                c.append(letter)
                print(len(c))
    cv.imshow("hi",img1)
    cv.imshow("hello",new_img)
    return c



model = tf.keras.models.load_model(r"C:\Users\Dell\Documents\GitHub\OCR\alpha_ker_datasets")
label = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}


img=cv.imread("test3.png")

scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation = cv.INTER_NEAREST)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur=cv.GaussianBlur(gray,(5,5),0)
cv.imwrite(r"images/test3.jpg",blur)
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
    if(i<2 or (i>=4 and i<6)):
        #str="_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        l[i]=cv.resize(l[i],(28,28),interpolation = cv.INTER_NEAREST)
        l[i]=l[i].reshape(1,28,28,1)
        data=label.get(np.argmax(model.predict(l[i]).flatten()))
    elif(i>=2 and i<4) or (i>=6 and i<10):
        str="_char_whitelist=0123456789"
        data=pytesseract.image_to_string(l[i], lang="eng", config=("--psm 10 ""--oem 2"" -c tessedit"+str+" -l osd"" "))
    ch.append(data)
print(*ch)



