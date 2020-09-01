import sys
sys.path.append('../')


from models import Resnet,VGG

def net(netname,num_classes,Train=True,Dataset=None):
    netname=netname.lower()
    if netname == "dnet34-34":
        return Resnet.Dnet34_34(num_classes)
    elif netname=="dnet50-34":
        return Resnet.Dnet50_34(num_classes)
    elif netname=="dnet34-18":
        return Resnet.Dnet34_18(num_classes)
    elif netname=="dnet18-18":
        return Resnet.Dnet18_18(num_classes)
    elif netname=="resnet18":
        return Resnet.Resnet18(num_classes)
    elif netname=="resnet34":
        return Resnet.Resnet34(num_classes)
    elif netname=="resnet50":
        return Resnet.Resnet50(num_classes)
    elif netname=="vgg19":
        return VGG.VGG19(num_classes)
    elif netname=="vgg16":
        return VGG.VGG16(num_classes)
    elif netname=="resnet50-vgg16":
        return VGG.Resnet50_VGG16(num_classes)
    else:
        raise RuntimeError('Unspported networks')
    

