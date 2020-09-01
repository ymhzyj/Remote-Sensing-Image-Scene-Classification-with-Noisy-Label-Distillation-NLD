from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.nn import init
class VGG19(nn.Module):
    def __init__(self, n_classes=45):
        super(VGG19, self).__init__()
        self.cleannet = nn.Sequential(
            models.vgg19(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(1000, n_classes)

    def forward(self, clean):

        c1 = self.cleannet(clean)
        c1 = self.fc1(c1)
        g = c1
        return g


class Resnet50_VGG16(nn.Module):
    def __init__(self, n_classes=45):
        super(Resnet50_VGG16, self).__init__()
        self.residualnet = nn.Sequential(
            models.resnet50(pretrained=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        self.cleannet = nn.Sequential(
            models.vgg16(pretrained=True),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(1000, n_classes)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, n_classes)

    def forward(self, noise, clean):

        r1 = self.residualnet(noise)
        r2 = self.cleannet(noise)
        c1 = self.cleannet(clean)
        c1 = self.fc1(c1)
        h = torch.cat((r1, r2), dim=1)
        h = self.fc2(h)
        h = self.fc3(h)
        g = c1
        return h, g


class VGG16(nn.Module):
    def __init__(self, n_classes=45):
        super(VGG16, self).__init__()
        self.cleannet = nn.Sequential(
            models.vgg16_bn(pretrained=True),
            nn.ReLU(inplace=True)
        )

        # for p in self.cleannet.parameters():
        #     p.requires_grad = False
        self.fc1 = nn.Linear(1000, n_classes)

    def forward(self, clean):

        c1 = self.cleannet(clean)
        c1 = self.fc1(c1)
        g = c1
        return g


if __name__ == '__main__':
    model = VGG16(n_classes=4)
    #input = t.autograd.Variable(t.randn(1, 3, 244, 244))
    input = torch.autograd.Variable(torch.randn(12, 3, 224, 224))
    output1 = model(input)
    print(output1.size())
