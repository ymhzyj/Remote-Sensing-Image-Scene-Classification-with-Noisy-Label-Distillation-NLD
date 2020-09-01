# from src import architectures,ramps
# import torch.backends.cudnn as cudnn
import csv
import os
import random
import time
import warnings
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

import losses.SCELoss as SCELoss
from dataset import ucmerced
from config.UCMerced.config_UCMerced import DefaultConfigs as config
from models import getnet
from utils import *

# from apex import amp
#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
# torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
obj=[
            "agricultural", "airplane", "baseballdiamond", "beach", " buildings", "chaparral", "denseresidential", "forest", "freeway", "golfcourse", "harbor",
            "intersection", "mediumresidential", "mobilehomepark", "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"
        ]
#3. test model on public dataset and save the probability matrix
def test(test_loader,model):
    top1 = AverageMeter(config)
    top5 = AverageMeter(config)
    matrix = runningScore(config=config)
    matrix.reset()
    times=0.0
    timeall =0.0
    precision1=0
    precision5=0
    #3.1 confirm the model converted to cuda
    # progress bar
    test_progressor = ProgressBar(mode="test",model_name=config.model_name, total=len(test_loader),weights=config.weights,Status=config.Status,current_time=config.time)
    # 2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            test_progressor.current = i
            image, target =sample['image'],sample['label'] 
            input = image.cuda()
            target = target.cuda()
            #target = Variable(target).cuda()
            # 2.2.1 compute output
            torch.cuda.synchronize()
            start = time.time()
            output= model(input)
            torch.cuda.synchronize()
            end = time.time()
            times=end-start
            timeall=timeall+times
            # output=output.squeeze(2)
            # output=output.squeeze(2)
        
            # 2.2.2 measure accuracy and record loss
            precision1, precision5 = accuracy(output, target, topk=(1, 5))
            matrix.update(output,target)
            top1.update(precision1[0],input.size(0))
            # top1.perclass(class_correct,class_total)
            top5.update(precision5[0], input.size(0))
            test_progressor.current_top1 = top1.avg
            test_progressor.current_top5 = top5.avg
            test_progressor()

            _, predicted = torch.max(output, 1) 


            # img_PIL=Image.open(origin_path).convert('RGB')
            # img_NP=np.array(img_PIL)
            # img_Tensor=torch.Tensor(img_NP)
      #img_Tensor=loader(img_PIL).unsqueeze(0)
            tag=obj[predicted.item()]
            right_label=obj[target.item()]
            resultdir=os.path.join(config.weights,config.model_name,config.Status,config.time)
            if os.path.exists( resultdir ):
                pass
            else:
                os.makedirs(resultdir)
            f=open(resultdir+'/upload.csv','a')
            csv_writer = csv.writer(f)
            csv_writer.writerow([right_label,tag])
            # img_path=resultdir+'/'+right_label+str(i)+'.png'
            # shutil.copy(str(origin_path),img_path)
        test_progressor.done()
        logdir = os.path.join(config.weights,config.model_name,config.Status,config.time)
        writer = SummaryWriter(logdir)
        confusion_matrix=matrix.get_value()
        np.save(logdir +'/confusion.npy',confusion_matrix)
        # writer.add_figure('confusion matrix',figure=plot_confusion_matrix(confusion_matrix, object_names=obj, title='Not Normalized confusion matrix',normalize=False,),global_step=1)
        writer.add_figure('confusion matrix',figure=plot_confusion_matrix(confusion_matrix, object_names=obj,title='Normalized confusion matrix',config=config,normalize=True),global_step=1)
        # fig=plot_confusion_matrix(confusion_matrix,obj,'Test Confusion_matrix')
        writer.close()
        precision,recall=matrix.get_scores()
        with open(os.path.join(config.weights,config.model_name,config.Status,config.time)+"/%s_test.txt"%config.model_name,"a") as f:
            for i in range(config.num_classes):
                print('Precision of %5s : %f %%' % (
                    obj[i], 100*precision[i]),file=f)
                print('Recall of %5s: %f%%'%(
                    obj[i], 100*recall[i]),file=f)
            print("Top1:%f,Top5:%f"%(top1.avg,top5.avg),file=f)
            print("avg Time:",timeall*1000/len(test_loader),"ms",file=f)
#4. more details to build main function    
def main():
    # fold = 0
    # #4.1 mkdirs
    # if not os.path.exists(config.weights):
    #     os.mkdir(config.weights)
    # if not os.path.exists(config.best_models):
    #     os.mkdir(config.best_models)
    # if not os.path.exists(config.logs):
    #     os.mkdir(config.logs)
    # if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
    #     os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    # if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
    #     os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)       
    #4.2 get model and optimizer
    model = getnet.net(config.model_name, config.num_classes,Dataset=config.dataset)
    #model = torch.nn.DataParallel(model)
    model.cuda()
    # model = amp.initialize(model, opt_level="O1") # 这里是“欧一”，不是“零一”
    #4.5 get files and split for K-fold dataset
    test_data_list = ucmerced.UCMerced_Clean(root=config.dataroot,split='test')
    test_dataloader = DataLoader(
        test_data_list, batch_size=1, shuffle=True,num_workers=config.workers, pin_memory=True)
    best_model = torch.load(os.path.join(config.weights,config.model_name,config.Status,config.time,'model_best.pth.tar'))
    model.load_state_dict(best_model["state_dict"])
    test(test_dataloader,model)

if __name__ =="__main__":
    # f=open(os.path.join('/home/pc-b3-218/Code/Cls/DNet/runs/CIFAR10/50000_balanced_labels/tree.txt'))
    # for i in range(14,15):
    #     config.data_seed=f.readline().strip()
    #     config.model_name=f.readline().strip()
    #     config.Status=f.readline().strip()
    #     config.time=f.readline().strip()
    #     # main()
    #     config.model_name=f.readline().strip()
    #     config.Status=f.readline().strip()
    #     config.time=f.readline().strip()
    #     main()
    # f.close()
    config_time=['2020-05-08-14-59-51','2020-05-08-15-42-10','2020-05-08-16-24-09','2020-05-08-17-06-00','2020-05-08-17-47-51']

    for i in range(5):
        config.time=config_time[i]
        main()
    # print("mean:{},std:{}"%np.mean(precisions),np.std(precisions))
        # sum+=precision1
    # avg=precision1/len(time_list)
    # print("avg precision1:%f"%avg.item())
