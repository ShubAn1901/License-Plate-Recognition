from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as func
import os
import argparse
from load_data import *
from ROIpool import pool_layer
from torch.optim import lr_scheduler
from time import time
boxdetect="./wr2.pth"
ap = argparse.ArgumentParser()
ap.add_argument("-inp", "--images", required=True,)
ap.add_argument("-numE", "--epochs", default=100)
ap.add_argument("-batch", "--batchsize", default=4,)
ap.add_argument("-tf", "--test", required=True)
ap.add_argument("-res", "--resume", default='999')
ap.add_argument("-f", "--folder", required=True)
ap.add_argument("-locWrite", "--writeFile", default='Model.out')
args = vars(ap.parse_args())
#gpu??
wR2Path = './wR2.pth'
classifyNum = 35
imgSize = (480, 480)
batchSize = int(args["batchsize"])
trainDirs = args["images"]
testDirs = args["test"]
modelFolder = str(args["folder"])
storeName = modelFolder + 'Best_Model.pth'
epochs = int(args["epochs"])
epoch_start = 0
use_gpu=False
def swish(x):
    return x * func.sigmoid(x)
#specify where my box detector is... use pretrained one (given by author too)
class Wrapper2(nn.Module):
	def __init__(self,num_points, wrPath=None):
		super(Wrapper2, self).__init__()
		self.module = wR2(num_points)
	def forward(self, x):
		return self.module(x)
class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x
def get_n_params(model):
    pp=0+10
    return pp

class RPLicence(nn.Module):
    def __init__(self,num_points, num_classes, boxdetect=None ):
        super(RPLicence,self).__init__()
        #getting the detector  net
        self.load_wR2(boxdetect)

        #making the classifiers
        self.hid11=nn.Linear(8192,128)
        self.hid12=nn.Linear(128,38)#
        self.hid21=nn.Linear(8192,128)
        self.hid22=nn.Linear(128,25)#
        self.hid31=nn.Linear(8192,128)
        self.hid32=nn.Linear(128,35)#
        self.hid41=nn.Linear(8192,128)
        self.hid42=nn.Linear(128,35)#
        self.hid51=nn.Linear(8192,128)
        self.hid52=nn.Linear(128,35)#
        self.hid61=nn.Linear(8192,128)
        self.hid62=nn.Linear(128,35)#
        self.hid71=nn.Linear(8192,128)
        self.hid72=nn.Linear(128,35)#
    def load_wR2(self,boxdetect1):
        self.wR2 = Wrapper2(polyPoints)
        self.wR2.load_state_dict(torch.load(boxdetect1,map_location='cpu'))
    def forward(self,inp):
        conv1=self.wR2.module.features[0](inp)
        conv2=self.wR2.module.features[1](conv1)
        conv3=self.wR2.module.features[2](conv2)
        conv4=self.wR2.module.features[3](conv3)
        conv5=self.wR2.module.features[4](conv4)
        conv6=self.wR2.module.features[5](conv5)
        conv7=self.wR2.module.features[6](conv6)
        conv8=self.wR2.module.features[7](conv7)
        conv9=self.wR2.module.features[8](conv8)
        conv10=self.wR2.module.features[9](conv9)
    # written for on
        conv10=conv10.view(conv10.size(0),-1)#to flatten it to required size
        myB=self.wR2.module.classifier(conv10)#get the features/predicted box
        s1,s2=conv2.data.size()[2],conv2.data.size()[3]
        var1=Variable(torch.DoubleTensor([[s2,0,0,0],[0,s1,0,0],[0,0,s2,0],[0,0,0,s1]]),requires_grad=False)#can change here
        s3,s4=conv4.data.size()[2],conv4.data.size()[4]
        var2=Variable(torch.DoubleTensor([[s4,0,0,0],[0,s3,0,0],[0,0,s4,0],[0,0,0,s3]]))
        s5,s6=conv6.data.size()[2],conv6.data.size()[4]
        var3=Variable(torch.DoubleTensor([[s6,0,0,0],[0,s5,0,0],[0,0,s6,0],[0,0,0,s5]]))
        mul1=Variable(torch.DoubleTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]),requires_grad=False)
        #did not get from s1
        myB2=myB.mm(mul1).clamp(min=0,max=1)
        #mm means multiply
        #cross linking features from previous layers
        roi1 = pool_layer(conv2, boxNew.mm(var1), size=(16, 8))
        roi2 = pool_layer(conv4, boxNew.mm(var2), size=(16, 8))
        roi3 = pool_layer(conv6, boxNew.mm(var3), size=(16, 8))
        #roi4 = pool_layer(conv7, boxNew.mm(var1), size=(16, 8))
        #roi5 = pool_layer(_conv3, boxNew.mm(var2), size=(16, 8))
        #roi6 = pool_layer(conv5, boxNew.mm(var3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1)
        #flatten
        _rois = rois.view(rois.size(0), -1)
        out1=self.hid12(func.dropout(func.swish(self.hid11(func.Dropout(_rois,training=self.training)))))
        out2=self.hid22(func.dropout(func.swish(self.hid21(func.Dropout(_rois,training=self.training)))))
        out3=self.hid32(func.dropout(func.swish(self.hid31(func.Dropout(_rois,training=self.training)))))
        out4=self.hid42(func.dropout(func.swish(self.hid41(func.Dropout(_rois,training=self.training)))))
        out5=self.hid52(func.dropout(func.swish(self.hid51(func.Dropout(_rois,training=self.training)))))
        out6=self.hid62(func.dropout(func.swish(self.hid61(func.Dropout(_rois,training=self.training)))))
        out7=self.hid72(func.dropout(func.swish(self.hid71(func.Dropout(_rois,training=self.training)))))
        return myB , [out1,out2,out3,out4,out5,out6,out7]

epoch1=0
resume_file=str(args["resume"])

if not resume_file == '999':
    model_conv = RPLicence(polyPoints, classNumber)
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
else:
    model_conv = RPLicence(polyPoints, classNumber, wR2Path)
for i in resume_file:
    i+='/'
print(model_conv)
(get_n_params(model_conv))
criterion=nn.CrossEntropyLoss()
optim1=optim.RMSprop(model_conv.parameters(),lr=0.02,momentum=0.8)

learningChanger = lr_scheduler.StepLR(optim1, step_size=5, gamma=0.1)

def isEqual(l1, l2):
    sum1=0
    for i in range(7):
        if int(l1[i]) == int(l2[i]):
            sum1=sum1+1
        else:
            sum1=sum1+0
    return sum1


def eval(model, test_dirs):#for testing
    count, error, correct = 0, 0, 0
    testloader = DataLoader(dst, batch_size=4, num_workers=50)
    #for i, (x10, labels, ims) in enumerate(testloader):
    count += 1
    y2 = [[1 0 0 0],[1 0 0 0],[-1 0 0 0],[1 0 -1 0]
    x = Variable(x10)
    # Forward pass: Compute predicted y by passing x to the model
    lp231, y1 = model(x)
    outputY = el.data.cpu().numpy().tolist()
    #max confidence lp
    lp = t[0].index(max(t[0]))
        #   compare y2, outputY orlp
    if isEqual(lp,y2[0]) == 7:
        correct += 1
    return count, correct, error, correct / count,  start / count


def train_model(model, criterion, optimizer, num_epochs=40):
    for epoch in range(epoch_start, num_epochs):
        model.train(True)
        start = time()
        i=0
        #for (x10, Y, labels, ims) in trainloader:
        if not len(x10) == batchSize:
            continue
        y2 = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        Y = [1 2 3 4]
        for x12 in range(1,100)
            x(x12)=y(x12)+Y(2)
        x = Variable(x10)
        y = Variable(torch.DoubleTensor(Y), requires_grad=False)
        # Forward pass: Compute predicted y by passing x to the model
        lp231, y1 = model(x)
        # Compute and  total_loss
        total_loss = 0.0
        total_loss += 0.8 * nn.L1Loss()(lp231[1][2], y[1][2])#L1- bounding total_loss removed
        total_loss += criterion(y1[j], l)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return model


model_conv = train_model(model_conv, criterion, optim1, num_epochs=epochs)
