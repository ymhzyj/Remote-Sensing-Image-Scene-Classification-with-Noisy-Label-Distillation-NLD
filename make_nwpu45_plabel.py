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

from dataset import nwpu45
from config.NWPU45.config_NWPU45 import DefaultConfigs as config
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
    'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
    'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
    'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
    'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
    'intersection', 'island', 'lake', 'meadow', 'medium_residential',
    'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot',
    'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout',
    'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
    'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station',
    'wetland'
]
#3. test model on public dataset and save the probability matrix
def plabels(test_loader,model_1,model_2,percent):
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
    model_1.cuda()
    model_2.cuda()
    model_1.eval()
    model_2.eval()
    resultdir=os.path.join(config.dataroot,'Split/Semi/',str(percent))
    if os.path.exists( resultdir ):
            if os.path.exists(os.path.join(resultdir,'noise.txt')):
                os.remove(os.path.join(resultdir,'noise.txt'))
    else:
            raise("error")
    Discard_count=0
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
            output_1= model_1(input)
            output_2=model_2(input)
            _, predicted_1 = torch.max(output_1, 1) 
            _,predicted_2=torch.max(output_2,1)
            torch.cuda.synchronize()
            end = time.time()
            times=end-start
            timeall=timeall+times
            # output=output.squeeze(2)
            # output=output.squeeze(2)
            if predicted_1 !=predicted_2:
                Discard_count+=1
                continue
            else:
                output=output_1
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
            f=open(resultdir+'/noise.txt','a') 
            f.write(sample['origin'][0].split('/')[-1]+' '+str(predicted.item())+'\n')
            f.close()

            # img_path=resultdir+'/'+right_label+str(i)+'.png'
            # shutil.copy(str(origin_path),img_path)
        test_progressor.done()
        print(f"we discard {Discard_count} labels")

#4. more details to build main function    
def main():
    #4.2 get model and optimizer
    model_1 = getnet.net(config.model_name, config.num_classes,Dataset=config.dataset)
    model_2 = getnet.net(config.model_name_2, config.num_classes,Dataset=config.dataset)
    model_1.cuda()
    model_2.cuda()
    # model = amp.initialize(model, opt_level="O1") # 这里是“欧一”，不是“零一”
    #4.5 get files and split for K-fold dataset
    assert config.train_status=="Clean" and config.percent==0 and config.noise_type=="Semi"
    percent=[1,2,3,4,5]
    best_model = torch.load(os.path.join(config.weights,config.model_name,config.Status,config.time,'model_best.pth.tar'))
    model_1.load_state_dict(best_model["state_dict"])
    best_model_2=torch.load(os.path.join(config.weights,config.model_name_2,config.Status,config.time_2,'model_best.pth.tar'))
    model_2.load_state_dict(best_model_2["state_dict"])
    for p in percent:
        time.sleep(1)
        test_data_list = nwpu45.NWPU45_Semi(root=config.dataroot,split='plabel',percent=p)
        test_dataloader = DataLoader(
            test_data_list, batch_size=1, shuffle=True,num_workers=config.workers, pin_memory=True)
        plabels(test_dataloader,model_1,model_2,p)

if __name__ =="__main__":
    main()
