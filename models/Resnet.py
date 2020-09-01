from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.nn import init
__all__ = ['Resnet18','Resnet34','Resnet50','Dnet50_34','Dnet34_34','Dnet34_18','Dnet18_18']

class Resnet18(nn.Module):
    def __init__(self, n_classes=45):
        super(Resnet18, self).__init__()
        self.cleannet=nn.Sequential(
            models.resnet34(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1= nn.Linear(1000,n_classes)

        
    def forward(self,clean):

        c1=self.cleannet(clean)
        c1=self.fc1(c1)
        g=c1
        return  g
class Resnet34(nn.Module):
    def __init__(self, n_classes=45):
        super(Resnet34, self).__init__()
        self.cleannet=nn.Sequential(
            models.resnet34(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1= nn.Linear(1000,n_classes)

        
    def forward(self,clean):

        c1=self.cleannet(clean)
        c1=self.fc1(c1)
        g=c1
        return  g

class Resnet50(nn.Module):
    def __init__(self, n_classes=45):
        super(Resnet50, self).__init__()
        self.cleannet=nn.Sequential(
            models.resnet50(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1= nn.Linear(1000,n_classes)

        
    def forward(self,clean):

        c1=self.cleannet(clean)
        c1=self.fc1(c1)
        g=c1
        return  g

class Dnet50_34(nn.Module):
    def __init__(self, n_classes=45):
        super(Dnet50_34, self).__init__()
        self.residualnet=nn.Sequential(
            models.resnet50(pretrained=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        self.cleannet=nn.Sequential(
            models.resnet34(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1= nn.Linear(1000,n_classes)
        self.fc2=nn.Linear(2000,1000)

        self.fc3= nn.Linear(1000,n_classes)

        
    def forward(self,noise,clean):

        r1=self.residualnet(noise)
        r2=self.cleannet(noise)
        c1=self.cleannet(clean)
        c1=self.fc1(c1)
        h=torch.cat( (r1,r2) ,dim = 1)
        h=self.fc2(h)
        h=self.fc3(h)
        g=c1
        return  h,g

class Dnet34_34(nn.Module):
    def __init__(self, n_classes=45):
        super(Dnet34_34, self).__init__()
        self.residualnet=nn.Sequential(
            models.resnet34(pretrained=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        self.cleannet=nn.Sequential(
            models.resnet34(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1= nn.Linear(1000,n_classes)
        self.fc2= nn.Linear(2000,1000)
        self.fc3= nn.Linear(1000,n_classes)

        
    def forward(self,noise,clean):

        r1=self.residualnet(noise)
        r2=self.cleannet(noise)
        c1=self.cleannet(clean)
        c1=self.fc1(c1)
        h=torch.cat( (r1,r2) ,dim = 1)
        h=self.fc2(h)
        h=self.fc3(h)
        g=c1
        return  h,g

class Dnet34_18(nn.Module):
    def __init__(self, n_classes=45):
        super(Dnet34_18, self).__init__()
        self.residualnet=nn.Sequential(
            models.resnet34(pretrained=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        self.cleannet=nn.Sequential(
            models.resnet18(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1= nn.Linear(1000,n_classes)
        self.fc2= nn.Linear(2000,1000)
        self.fc3= nn.Linear(1000,n_classes)

        
    def forward(self,noise,clean):

        r1=self.residualnet(noise)
        r2=self.cleannet(noise)
        c1=self.cleannet(clean)
        c1=self.fc1(c1)
        h=torch.cat( (r1,r2) ,dim = 1)
        h=self.fc2(h)
        h=self.fc3(h)
        g=c1
        return  h,g

class Dnet18_18(nn.Module):
    def __init__(self, n_classes=45):
        super(Dnet18_18, self).__init__()
        self.residualnet=nn.Sequential(
            models.resnet18(pretrained=True),            
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        self.cleannet=nn.Sequential(
            models.resnet18(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1= nn.Linear(1000,n_classes)
        self.fc2= nn.Linear(2000,1000)
        self.fc3= nn.Linear(1000,n_classes)

        
    def forward(self,noise,clean):

        r1=self.residualnet(noise)
        r2=self.cleannet(noise)
        c1=self.cleannet(clean)
        c1=self.fc1(c1)
        h=torch.cat( (r1,r2) ,dim = 1)
        h=self.fc2(h)
        h=self.fc3(h)
        g=c1
        return  h,g
if __name__=='__main__':
        model =Dnet34_34(n_classes=4)
        #input = t.autograd.Variable(t.randn(1, 3, 244, 244))
        input = torch.autograd.Variable(torch.randn(2,3,32,32))
        output1,output2=model(input,input)
        print(output1.size())