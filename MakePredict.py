#encoding:utf-8
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import argparse
import numpy as np
from os import path, mkdir
from load_data import *
from time import time
from ROIpool import pool_layer
class WrappedModel(nn.Module):
	def __init__(self,num_points, num_classes, wrPath=None):
		super(WrappedModel, self).__init__()
		self.module = RPlicence(num_points,num_classes) # that I actually define.
	def forward(self, x):
		return self.module(x)
class WrappedModel2(nn.Module):
	def __init__(self,num_points, wrPath=None):
		super(WrappedModel2, self).__init__()
		self.module = wR2(num_points) # that I actually define.
	def forward(self, x):
		return self.module(x)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,)
ap.add_argument("-m", "--model", required=True,)
args = vars(ap.parse_args())

use_gpu =False
#print (use_gpu)

cnum = 4
nPt = 4
imgSize = (480, 480)
batchSize = 8 if use_gpu else 8
resume_file = str(args["model"])

provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
english = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
numEng = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

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


class RPlicence(nn.Module):
    def __init__(self, num_points, num_classes, wrPath=None):
        super(RPlicence, self).__init__()
        self.load_wR2(wrPath)
        self.classifier1 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, provNum),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphaNum),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )

    def load_wR2(self, path):
        self.wR2 = WrappedModel2(nPt)#wR2(nPt)
        #self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None :
            self.wR2.load_state_dict(torch.load(path,map_location='cpu'))
            # self.wR2 = self.wR2.cuda()
        # for param in self.wR2.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)

        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        var1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]), requires_grad=False)
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        var2= Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]), requires_grad=False)
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        var3= Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]), requires_grad=False)
        postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        # input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
        # zs = Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]), requires_grad=False)
        z1 = pool_layer(_x1, boxNew.mm(var1), size=(16, 8))
        z2 = pool_layer(_x3, boxNew.mm(var2), size=(16, 8))
        z3 = pool_layer(_x5, boxNew.mm(var3), size=(16, 8))
        zs = torch.cat((z1, z2, z3), 1)
        flatROI=zs.view(zs.size(0), -1)
		#final characters
        out1=self.classifier1(flatROI)
        out2=self.classifier2(flatROI)
        out3=self.classifier3(flatROI)
        out4=self.classifier4(flatROI)
        out5=self.classifier5(flatROI)
        out6=self.classifier6(flatROI)
        out7=self.classifier7(flatROI)
        return boxLoc,[out1, out2, out3, out4, out5, out6, out7]


def isEqual(labelGT, labelP):
    print (labelGT)
    print (labelP)
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)


model1 = WrappedModel(nPt,cnum)
model1.load_state_dict(torch.load(resume_file,map_location='cpu'))
model1.eval()


tdataloader = demoTestDataLoader(args["input"].split(','), imgSize)
trainloader = DataLoader(tdataloader, batch_size=1, num_workers=1)
#loop to iterate over all
for i, (XI, ims) in enumerate(trainloader):
	x = Variable(XI)
	box,y_pred = model1(x)
	#this is their syntax, we have used this as it is really quick
	outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
	labelPred = [t[0].index(max(t[0])) for t in outputY]
	[boxcentX, boxcentY, w, h] = box.data.cpu().numpy()[0].tolist()
	img = cv2.imread(ims[0])
	luCorner = [(boxcentX - w/2)*img.shape[1], (boxcentY - h/2)*img.shape[0]]
	rdCorner = [(boxcentX + w/2)*img.shape[1], (boxcentY + h/2)*img.shape[0]]
	cv2.rectangle(img, (int(luCorner[0]), int(luCorner[1])), (int(rdCorner[0]), int(rdCorner[1])), (240, 30, 25), 4)
	lpn = english[labelPred[1]] + numEng[labelPred[2]] + numEng[labelPred[3]] + numEng[labelPred[4]] + numEng[labelPred[5]] + numEng[labelPred[6]]
	cv2.putText(img, lpn, (int(luCorner[0]), int(luCorner[1])-30), cv2.FONT_ITALIC, 3, (40, 255, 35),3)
	cv2.imwrite(ims[0], img)
