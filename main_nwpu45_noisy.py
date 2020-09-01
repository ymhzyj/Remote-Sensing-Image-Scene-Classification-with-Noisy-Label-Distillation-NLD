import multiprocessing
import os
# from apex import amp
import pickle
import random
import re
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

import losses.SCELoss as SCELoss
from config.NWPU45.config_NWPU45 import DefaultConfigs as config
from dataset import nwpu45
from models import getnet
from utils import *

multiprocessing.set_start_method('spawn',True)
# import torch.backends.cudnn as cudnn
# 1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
# torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
# 2. evaluate func

def evaluate(val_loader, model,criterion, epoch):
    # 2.1 define meters
    losses = AverageMeter(config=config)
    top1 = AverageMeter(config=config)
    top5 = AverageMeter(config=config)
    # progress bar
    val_progressor = ProgressBar(mode="Val  ", epoch=epoch, total_epoch=config.epochs,
                                 model_name=config.model_name, total=len(val_loader),weights=config.weights,Status=config.Status,current_time=config.time)
    # 2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    # model.shake_config=(True,False,True)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            val_progressor.start_time=time.time()
            # image, target = sample['image'], sample['label']
            val_progressor.current = i
            image, target =sample['image'],sample['label'] 
            input2_size = image.size()
            input2 = np.zeros(input2_size).astype(np.float32)
            input2 = torch.from_numpy(input2).cuda()
            input = image.cuda()
            target = target.cuda()
            # 2.2.1 compute output
            _,output= model(input2,input)
            loss = criterion(output, target)

            # 2.2.2 measure accuracy and record loss
            precision1, precision5 = accuracy(output, target, topk=(1, 5))
            class_correct, class_total=perclass_precision(output, target,config)
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            top1.perclass(class_correct,class_total)
            top5.update(precision5[0], input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor.current_top5 = top5.avg
            val_progressor.end_time=time.time()
            val_progressor()
        val_progressor.done()
    return [losses.avg, top1.avg, top5.avg,top1.perclass_avg]

def main():
    # 4.1 tensorboard
    current_time = time.strftime(
        '%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    config.time=current_time
    model = getnet.net(config.model_name, config.num_classes,Train=True,Dataset=config.dataset)

    model.cuda()
    # model=torch.nn.DataParallel(model)
    # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    optimizer = optim.Adam(model.parameters(),lr = config.lr,amsgrad=True,weight_decay=config.weight_decay)
    criterion_clean = nn.CrossEntropyLoss().cuda()
    criterion_noise = nn.CrossEntropyLoss().cuda()
    # criterion_noise=SCELoss.SCELoss(alpha=0.1, beta=1, num_classes=config.num_classes)
    # 4.3 some parameters for  K-fold and restart model
    start_epoch = 0
    best_precision1 = 0
    best_precision5 = 0
    # best_precision_save = 0
    loss_status=""
    early_stopping = EarlyStopping(patience=config.patience, verbose=True,current_time=current_time,config=config)
    # 4.4 restart the training process
    if config.Finetune:
        checkpoint = torch.load(os.path.join(config.weights,config.model_name,config.Status,config.time , 'model_best.pth.tar'))
        start_epoch = checkpoint["epoch"]
        current_time = checkpoint["current_time"]
        best_precision1 = checkpoint["best_precision1"]
        best_precision5 = checkpoint["best_precision5"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_status=checkpoint["loss"]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # mkdir
    if not os.path.exists(config.weights):
        os.makedirs(config.weights)
    if not os.path.exists(os.path.join(config.weights,config.model_name,config.Status,current_time)):
        os.makedirs(os.path.join(config.weights,config.model_name,config.Status,current_time))
    logdir = os.path.join(config.weights, config.model_name, config.Status,current_time)
    writer = SummaryWriter(logdir)
    shutil.copyfile('./config/'+config.dataset+'/config_'+config.dataset+'.py', os.path.join(config.weights,config.model_name,config.Status ,str(current_time),'config.py'))
    
    if config.noise_type=="None":
        assert config.train_status=="Double" and config.percent==0
        train_data_list=nwpu45.NWPU45_Clean(root=config.dataroot,split='train')
    elif config.noise_type=="Asym" or config.noise_type=="Symm" or config.noise_type=="Semi":
        assert config.train_status=="Double" and config.percent !=0
        train_data_list=nwpu45.NWPU45_Noise_Train(root=config.dataroot,config=config)
    else:
        raise("unsupport noise_type or train_status")

    val_data_list=nwpu45.NWPU45_Clean(root=config.dataroot,split='val')

    train_dataloader = DataLoader(
        train_data_list, batch_size=config.noisebatch_size, num_workers=config.workers,shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(
        val_data_list, batch_size=config.batch_size, shuffle=True,num_workers=config.workers, pin_memory=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)
    # scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,5) #2
    # scheduler=optim.lr_scheduler.MultiStepLR(optimizer,[40,80])
    # scheduler=optim.lr_scheduler.ExponentialLR(optimizer,0.1 , last_epoch=-1)

    # 4.5.5.1 define metrics
    train_losses = AverageMeter(config=config)
    train_clean_losses= AverageMeter(config=config)
    train_clean_top1 = AverageMeter(config=config)
    train_clean_top5 = AverageMeter(config=config)
    train_noise_top1 = AverageMeter(config=config)
    train_noise_top5 = AverageMeter(config=config)
    valid_loss = [np.inf, 0, 0]
    # model.train()

    # 4.5.5 train

    for epoch in range(start_epoch, config.epochs):
        avg_loss=0
        scheduler.step(epoch)
        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=config.epochs,
                                       model_name=config.model_name, total=len(train_dataloader),weights=config.weights,Status=config.Status,current_time=current_time)
        # cleaniter=ir(train_dataloader)
        # train
        #global ir
        for ir,  (sample) in enumerate(train_dataloader):
            # 4.5.5 switch to continue train process
            if len(sample)==3:
                clean_image,clean_target,noise_image,noise_target=sample['image'],sample['label'],sample['image'],sample['label']
            elif len(sample)==2:
                clean_image,clean_target,noise_image,noise_target=sample[0]['image'],sample[0]['label'],sample[1]['image'],sample[1]['label']
            train_progressor.start_time=time.time()
            train_progressor.current = ir
            global_iter = len(train_dataloader) * epoch + ir + 1
            model.train()
            # model.shake_config=(True, True, True)
            clean_image = clean_image.cuda()
            clean_target = clean_target.cuda()
            noise_image = noise_image.cuda()
            noise_target = noise_target.cuda()           
            h,g = model(noise_image,clean_image)
            noise_loss = criterion_noise(h, noise_target)
            clean_loss = criterion_clean(g, clean_target)
            loss = 10*clean_loss+2*noise_loss
            loss_status = "10*clean_loss+2*noise_loss"
            precision1_noise_train, precision5_noise_train = accuracy(h, noise_target, topk=(1, 5))
            precision1_clean_train, precision5_clean_train = accuracy(g, clean_target, topk=(1, 5))
            train_losses.update(loss.item(), clean_image.size(0)+noise_image.size(0))
            train_clean_losses.update(clean_loss.item(), clean_image.size(0))
            class_correct, class_total=perclass_precision(g, clean_target,config)
            train_clean_top1.perclass(class_correct,class_total)
            train_clean_top1.update(precision1_clean_train[0], clean_image.size(0))
            train_clean_top5.update(precision5_clean_train[0], clean_image.size(0))
            train_noise_top1.update(precision1_noise_train[0], noise_image.size(0))
            train_noise_top5.update(precision5_noise_train[0],noise_image.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_clean_top1.avg
            train_progressor.current_top5 = train_clean_top5.avg
            current_top1=train_progressor.current_top1
            current_noise_top5 = train_noise_top5.avg
            current_noise_top1 = train_noise_top1.avg
            writer.add_scalar(
                'train/clean_top5', train_progressor.current_top5, global_iter)
            writer.add_scalar(
                'train/clean_top1', train_progressor.current_top1, global_iter)
            writer.add_scalar(
                'train/noise_top5', current_noise_top5, global_iter)
            writer.add_scalar(
                'train/noise_top1', current_noise_top1, global_iter)
            # backward

            optimizer.zero_grad()
            avg_loss+=loss.item()
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            writer.add_scalar('train/total_loss_iter',
                              loss.item(), global_iter)
            writer.add_scalar('train/total_clean_loss_iter',
                               clean_loss.item(), global_iter)
            writer.add_scalar('train/total_noise_loss_iter',
                               noise_loss.item(), global_iter)
            train_progressor.end_time=time.time()
            train_progressor()

        train_progressor.done()
        writer.add_scalar('train/avg_loss_epochs',
                              avg_loss/len(train_dataloader), epoch)
        #end = time.clock()
        # evaluate
        lr = get_learning_rate(optimizer)
        
        writer.add_scalar('parameters/learning_rate',lr,epoch)
        # evaluate every half epoch
        
        valid_loss = evaluate(val_dataloader, model, criterion_clean,epoch)
        writer.add_scalar('val/top1', valid_loss[1], epoch)
        writer.add_scalar('val/top5', valid_loss[2], epoch)
        is_best1 = valid_loss[1] > best_precision1
        is_best5 = valid_loss[2] >best_precision5
        best_precision1 = max(valid_loss[1], best_precision1)
        best_precision5 = max(valid_loss[2], best_precision5)
        perclass=valid_loss[3]
        #Early
        early_stopping(val_loss=valid_loss[0],state={
            "epoch": epoch + 1,
            "model_name": config.model_name,
            "state_dict": model.state_dict(),
            "best_precision1": best_precision1,
            "best_precision5": best_precision5,
            "perclass":perclass,
            "optimizer": optimizer.state_dict(),
            "current_time": current_time,
            "valid_loss": valid_loss,
            "loss":  loss_status
        }, is_best1=is_best1)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # try:
        #     best_precision_save = best_precision1.cpu().data.numpy()
        #     best_precision_save = best_precision5.cpu().data.numpy()
        # except:
        #     pass
        # save_checkpoint()


if __name__ == "__main__":

    main()
