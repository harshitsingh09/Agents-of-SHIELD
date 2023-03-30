import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import models
import torch.nn as nn
import torch
import fastai as Path
from torch.autograd import Variable

nos=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphas=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
all_char=nos+alphas

def encode(a):
    onehot=[0]*len(all_char)
    onehot[all_char.index(a)]+=1
    return onehot
    
class my_data(Dataset):
    def __init__(self,path,isTrain=True,transform=None):
        self.path=path
        if isTrain:
            self.img=os.listdir(self.path)[:25]
        else:
            self.img=os.listdir(self.path)[26:]
        self.transform=transform
    
    def __getitem__(self,idx):
        self.img_name=self.img[idx]
        img=Image.open(self.path+"/"+self.img_name)
        img=img.convert("L")
        label=[]
        for i in self.img_name[:-4]:
            label+=(encode(i))
        if self.transform is not None:
            img=self.transform(img)
        return img,np.array(label),label
    
    def __len__(self):
        return len(self.img)    

transform=transforms.Compose([transforms.Resize((224,224),interpolation=2),transforms.ToTensor()])#normalize

t_ds=my_data(r"C:\Users\Dell\Documents\GitHub\anpr\images",transform=transform)
test=my_data(r"C:\Users\Dell\Documents\GitHub\anpr\images",False,transform=transform)
t_dl=DataLoader(t_ds,64,0)
tes=DataLoader(test,1,0)

model=models.resnet18(weights=None)
model.conv1=nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
model.fc = nn.Linear(in_features=512, out_features=144, bias=True)

loss_func = nn.MultiLabelSoftMarginLoss()
optm = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for step,i in enumerate(t_dl):
        img,label,_ = i
        img = Variable(img)
        pred = model(img)
        label=Variable(label.float())
        print(pred.size())
        print(label.size())
        loss = loss_func(pred,label)
        optm.zero_grad()
        loss.backward()
        optm.step()
        print('eopch:', epoch+1, 'step:', step+1, 'loss:', loss.item())


 
