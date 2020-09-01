import shutil
import torch

import os

import numpy as np
import torchnet as tnt

import matplotlib.pyplot as plt
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,config,patience=100, verbose=False, delta=0,current_time=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.current_time = current_time
        self.config=config

    def __call__(self, val_loss,state, is_best1):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,state, is_best1,True)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_checkpoint(val_loss,state, is_best1,False)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,state, is_best1,True)
            self.counter = 0

    def save_checkpoint(self, val_loss,state, is_best1,is_decdreased):
        config=self.config
        '''Saves model when validation loss decrease.'''
        if is_decdreased:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            filename = os.path.join(config.weights,config.model_name,config.Status ,str(self.current_time) , "_checkpoint.pth.tar")
            torch.save(state, filename)
        if is_best1:
            message = os.path.join(config.weights,config.model_name,config.Status ,str(self.current_time) , 'model_best.pth.tar')
            print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"],message))
            with open(os.path.join(config.weights,config.model_name,config.Status,str(self.current_time),"%s.txt"%config.model_name),"a") as f:
                print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"],message),file=f)
                for i in range(config.num_classes):
                    print('Precision of %5s : %f %%' % (
                        i, 100*state["perclass"][i].cpu()),file=f)
            torch.save(state, message)
        self.val_loss_min = val_loss

# def save_checkpoint(state, is_best1,current_time):
#     filename = os.path.join(config.weights,config.model_name,config.Status ,str(current_time) , "_checkpoint.pth.tar")
#     torch.save(state, filename)
#     if is_best1:
#         message = os.path.join(config.weights,config.model_name,config.Status ,str(current_time) , 'model_best.pth.tar')
#         print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"],message))
#         with open(config.weights+config.model_name+ os.sep+config.Status+os.sep+str(current_time)+os.sep +"%s.txt"%config.model_name,"a") as f:
#             print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"],message),file=f)
#             for i in range(config.num_classes):
#                 print('Precision of %5s : %f %%' % (
#                     i, 100*state["perclass"][i].cpu()),file=f)
#         shutil.copyfile(filename, message)
    # if is_best5:
    #     message = config.best_models + config.model_name+ os.sep +str(current_time)  + os.sep + 'model_best_5.pth.tar'
    #     print("Get Better top5 : %s saving weights to %s"%(state["best_precision5"],message))
    #     with open("./logs/%s.txt"%config.model_name,"a") as f:
    #         print("Get Better top5 : %s saving weights to %s"%(state["best_precision5"],message),file=f)
    #     shutil.copyfile(filename, message)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self,config):
        self.config=config
        self.reset()

    def reset(self):
        config=self.config
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.class_correct = list(0. for i in range(config.num_classes))
        self.class_total = list(0. for i in range(config.num_classes))
        self.perclass_avg= list(0. for i in range(config.num_classes))

    def update(self, val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def perclass(self,class_correct,class_total):
        config=self.config
        for i in range(config.num_classes):
            self.class_correct[i]+=class_correct[i]
            self.class_total[i]+=class_total[i]
            if class_total[i]!=0:
                self.perclass_avg[i]=self.class_correct[i]/self.class_total[i]
# def adjust_learning_rate(optimizer, epoch,config):
#     """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
#     if epoch==5:
#         lr=config.lr_2
#     elif epoch==10:
#         lr=config.lr_3
#     # lr = config.lr * (0.1 ** (epoch // 3))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def adjust_learning_rate(optimizer, lr,epoch):
    if epoch<150:
        lr=lr
    elif epoch<250:
        lr=lr*0.1
    else:
        lr=lr*0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def schedule(current_epoch, current_lrs, **logs):
        lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
        epochs = [0, 1, 6, 8, 12]
        for lr, epoch in zip(lrs, epochs):
            if current_epoch >= epoch:
                current_lrs[5] = lr
                if current_epoch >= 2:
                    current_lrs[4] = lr * 1
                    current_lrs[3] = lr * 1
                    current_lrs[2] = lr * 1
                    current_lrs[1] = lr * 1
                    current_lrs[0] = lr * 0.1
        return current_lrs

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def perclass_precision(output,target,config):
    with torch.no_grad():
        batch_size = target.size(0)
        # pred=output
        _, pred = torch.max(output, 1)
        correct_class = (pred == target).squeeze().float()
        class_correct= list(0. for i in range(config.num_classes))
        class_total= list(0. for i in range(config.num_classes))
        if batch_size>1:
            for i in range(batch_size):
                class_correct[target[i]] +=  correct_class[i]
                class_total[target[i]]+=1
        else:
                class_correct[target] +=  correct_class
                class_total[target]+=1
    return class_correct, class_total

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError


class runningScore(object):
    def __init__(self,config):
            self.confusion_matrix=tnt.meter.ConfusionMeter(config.num_classes)
    def update(self, pred, target):
            self.confusion_matrix.add(pred,target)
    def get_value(self):
            return self.confusion_matrix.value()
    def get_scores(self):
            M=self.confusion_matrix.value()
            n=len(M)
            precision= list(0. for i in range(n))
            recall= list(0. for i in range(n))
            for i in range(len(M[0])):
                rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
                if rowsum!=0:
                    precision[i]=M[i][i]/float(rowsum)
                else:
                    precision[i]=0
                if colsum!=0:
                    recall[i]=M[i][i]/float(colsum) 
                else:
                    recall[i]=0

            return precision, recall

    def reset(self):
            self.confusion_matrix.reset()
def plot_confusion_matrix(cm,object_names, title,config,normalize=False):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
        else:
            cm=cm.astype('float')
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
        plt.title(title)    # 图像标题
        plt.colorbar()
        num_local = np.array(range(config.num_classes))    
        plt.xticks(num_local, object_names, rotation=90)    # 将标签印在x轴坐标上
        plt.yticks(num_local, object_names)    # 将标签印在y轴坐标上
        plt.tight_layout()
        plt.ylabel('True label')    
        plt.xlabel('Predicted label')
        return fig




