from django.shortcuts import render
from django.http import HttpResponse
import cv2 as cv
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create your views here.

def sort_contours(cnts):
    i = 0
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i]))
    return cnts

def img_check(new_img,img1):
    contours,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    c=[]
    for cnt in sort_contours(contours):
        x,y,w,h=cv.boundingRect(cnt)
        ratio=h/w
        if(ratio>=1 and ratio<=4.6):
            if(h/len(new_img)>=0.5):
                cv.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),1)
                letter=new_img[y:y+h,x:x+w]
                c.append(letter)
    cv.imshow("hi",new_img)
    return c
embedder=FaceNet()
def get_embeddings(faces):

    faces = faces.astype('float32')
    faces = np.expand_dims(faces, axis=0)
    yhat = embedder.embeddings(faces)
    return yhat[0]

def index(request):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    img=cv.imread(r"C:\Users\nithe\OneDrive\Desktop\kavach_hackathon\Images\test.png")
    img=cv.resize(img,(524,225),interpolation = cv.INTER_AREA)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(gray,(7,7),0)
    _,binary=cv.threshold(blur,180,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    new_img=cv.morphologyEx(binary,cv.MORPH_DILATE,cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
    contours,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    l=[]
    for cnt in contours:
        x,y,w,h=cv.boundingRect(cnt)
        if(h*w>150*500):
            l1=img_check(new_img[0:255//2,:],img[0:255//2,:])
            l2=img_check(new_img[255//2-20:255,:],img[255//2:255,:])
            l=l1.append(l2)
            
        else:
            l=img_check(new_img,img)

            
    erd = cv.erode(new_img, None, iterations=2)
    str="_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    data=pytesseract.image_to_string(erd, lang="eng", config=("--psm 13 ""--oem 1"" -c tessedit"+str+" -l osd"" "))
    ch=""
    print(data)
    for i in range(len(l)):
        erd = cv.erode(l[i], None, iterations=2)
        if(i<2 or (i>=4 and i<6)):
            str="_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif(i>=2 and i<4) or (i>=6 and i<10):
            str="_char_whitelist=0123456789"
        data=pytesseract.image_to_string(erd, lang="eng", config=("--psm 10 ""--oem 2"" -c tessedit"+str+" -l osd"" "))
        ch+=data.split("\n")[0]


    img = cv.imread(r"C:\Users\nithe\OneDrive\Desktop\kavach_hackathon\dumb\WhatsApp Image 2023-03-31 at 15.39.29.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    detector = MTCNN()
    results = detector.detect_faces(img)
    x,y,w,h = results[0]['box']
    img = cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)
    plt.imshow(img)
    my_face = img[y:y+h, x:x+w]
    #Facenet takes as input 160x160 
    my_face = cv.resize(my_face, (160,160))
    plt.imshow(my_face)
    faceloading = FACELOADING(r"C:\Users\nithe\OneDrive\Desktop\kavach_django\lp_predictor\static\test folder")
    X, Y = faceloading.load_classes()
    plt.figure(figsize=(16,12))
    for num,image in enumerate(X):
        ncols = 3
        nrows = len(Y)//ncols + 1
        plt.subplot(nrows,ncols,num+1)
        plt.imshow(image)
        plt.axis('off')
    EMBEDDED_X = []
    for face in X:
        EMBEDDED_X.append(get_embeddings(face))
    EMBEDDED_X = np.asarray(EMBEDDED_X)
    np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    plt.plot(EMBEDDED_X[0]) 
    plt.ylabel(Y[0])
    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)
    ypreds_train = model.predict(X_train)
    ypreds_test = model.predict(X_test)
    accuracy_score(Y_train, ypreds_train)
    accuracy_score(Y_test,ypreds_test)
    img = cv.imread(r"C:\Users\nithe\OneDrive\Desktop\kavach_django\lp_predictor\static\download.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    x,y,w,h = detector.detect_faces(img)[0]['box']
    img = img[y:y+h, x:x+w]
    img = cv.resize(img, (160,160))
    test_im = get_embeddings(img) 
    test_im = [test_im]
    ypreds = model.predict(test_im)
    y=encoder.inverse_transform(ypreds)[0]
    return render(request,"C://Users//nithe//OneDrive//Desktop//kavach_django//lp_predictor//members//templates//result.html",context={'result':ch,'faceresult':y})
class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()
    

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr
    

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        
        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')
# def index(request):
    
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def sort_contours(cnts):
#     i = 0
#     boundingBoxes = [cv.boundingRect(c) for c in cnts]
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i]))
#     return cnts

# def img_check(new_img,img1):
#     contours,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
#     c=[]
#     for cnt in sort_contours(contours):
#         x,y,w,h=cv.boundingRect(cnt)
#         ratio=h/w
#         if(ratio>=1 and ratio<=4.6):
#             if(h/len(new_img)>=0.5):
#                 cv.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),1)
#                 letter=new_img[y:y+h,x:x+w]
#                 c.append(letter)
#     cv.imshow("hi",new_img)
#     return c


#     img=cv.imread("test.png")
#     img=cv.resize(img,(524,225),interpolation = cv.INTER_AREA)
#     gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     blur=cv.GaussianBlur(gray,(7,7),0)
#     _,binary=cv.threshold(blur,180,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
#     new_img=cv.morphologyEx(binary,cv.MORPH_DILATE,cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
#     contours,_=cv.findContours(new_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
#     l=[]
#     for cnt in contours:
#         x,y,w,h=cv.boundingRect(cnt)
#         if(h*w>150*500):
#             l1=img_check(new_img[0:255//2,:],img[0:255//2,:])
#             l2=img_check(new_img[255//2-20:255,:],img[255//2:255,:])
#             l=l1.append(l2)
            
#         else:
#           l=img_check(new_img,img)

        
# fig = plt.figure(figsize=(14,4))
# grid = gridspec.GridSpec(ncols=len(l),nrows=1,figure=fig)
# cv.resize(l[0],(3000,3000),interpolation = cv.INTER_AREA)
# erd = cv.erode(new_img, None, iterations=2)
# str="_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# data=pytesseract.image_to_string(erd, lang="eng", config=("--psm 13 ""--oem 1"" -c tessedit"+str+" -l osd"" "))
# ch=""
# print(data)
# for i in range(len(l)):
#     erd = cv.erode(l[i], None, iterations=2)
#     if(i<2 or (i>=4 and i<6)):
#         str="_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     elif(i>=2 and i<4) or (i>=6 and i<10):
#         str="_char_whitelist=0123456789"
#     data=pytesseract.image_to_string(erd, lang="eng", config=("--psm 10 ""--oem 2"" -c tessedit"+str+" -l osd"" "))
#     ch+=data




# return render(request,'C://Users//nithe//OneDrive//Desktop//kavach_django//lp_predictor//members//templates//result.html')
#    pytesseract.pytesseract.tesseract_cmd='C://Program Files//Tesseract-OCR//tesseract.exe'
#    image=cv2.imread('C://Users//nithe//OneDrive//Desktop//kavach_hackathon//Images//8.jpg')
#    image=imutils.resize(image,width=300)
#    cv2.imshow("original image",image)
#    cv2.waitKey(0)
#    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    cv2.imshow("greyed image",gray_image)
#    cv2.waitKey(0)
#    gray_image=cv2.bilateralFilter(gray_image,11,17,17)
#    cv2.imshow("smoothened image",gray_image)
#    cv2.waitKey(0)
#    edged=cv2.Canny(gray_image,30,200)
#    cv2.imshow("edged image",edged)
#    cv2.waitKey(0)
#    cnts,new=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#    image1=image.copy()
#    cv2.drawContours(image1,cnts,-1,(0,255,0),3)
#    cv2.imshow("countours",image1)
#    cv2.waitKey(0)
#    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:50]
#    screenCnt=None
#    image2=image.copy()
#    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
#    cv2.imshow("Top 30 contours",image2)
#    cv2.waitKey(0)
#    cv2.waitKey(0)
#    i=7
#    for c in cnts:
#         perimeter=cv2.arcLength(c,True)
#         approx=cv2.approxPolyDP(c,0.018*perimeter,True)
#         if len(approx)==4:
#             screenCnt=approx
#             x,y,w,h=cv2.boundingRect(c)
#             new_img=image[y:y+h,x:x+w]
#             cv2.imwrite('C://Users//nithe//OneDrive//Desktop//kavach_hackathon//'+str(i)+'.jpg',new_img)
#             i+=1
#             break
        
#    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
#    cv2.imshow("image with detected license plate", image)
#    cv2.waitKey(0)

#    Cropped_loc = 'C://Users//nithe//OneDrive//Desktop//kavach_hackathon//7.jpg'
#    final_image=cv2.imread(Cropped_loc)
#    cv2.imshow("cropped",final_image)
#    plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
#    print("Number plate is:", plate)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
   
