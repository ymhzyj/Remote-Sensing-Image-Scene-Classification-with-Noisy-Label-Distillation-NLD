import sys 
import re 
import os
import time
class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)1d%%"
    def __init__(self,mode,epoch=0,total_epoch=None,current_loss=None,current_top1=None,current_top5=None,model_name=None,total=None,current=None,weights=None,Status=None,current_time=None,width = 35,symbol = ".",output=sys.stderr):
        assert len(symbol) == 1

        self.mode = mode
        self.total = total
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.current_loss = current_loss
        self.current_top1 = current_top1
        self.current_top5 = current_top5
        self.model_name = model_name
        self.weights= weights
        self.Status=Status
        self.current_time = current_time
        self.start_time = 0
        self.end_time = 0
        self.spend_time = 0
    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"
        speed=self.end_time-self.start_time
        self.spend_time += speed
        args = {
            "mode":self.mode,
            "total": self.total,
            "bar" : bar,
            "current": self.current,
            "percent": percent * 100,
            "current_loss":self.current_loss,
            "current_top1":self.current_top1,
            "current_top5":self.current_top5,
            "epoch":self.epoch + 1,
            "epochs":self.total_epoch,
            "spend_time":self.spend_time,
            "total_time":speed*self.total
        }
        if self.mode != "test":
            message = "\033[1;32;40m%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s\033[0m  [ Loss %(current_loss).5f  Top1: %(current_top1).2f Top5: %(current_top5).2f]  %(current)d/%(total)d time:%(spend_time).2fs/%(total_time).2fs \033[1;32;40m[ %(percent)3d%% ]\033[0m" %args
            self.write_message = "%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s  [ Loss %(current_loss).5f Top1: %(current_top1).2f Top5: %(current_top5).2f]  %(current)d/%(total)d [ %(percent)3d%% ]" %args
            print("\r" + message,file=self.output,end="")
        else:
            message = "\033[1;32;40m%(mode)s  %(bar)s \033[0m  [Top1: %(current_top1).2f Top5: %(current_top5).2f]  %(current)d/%(total)d \033[1;32;40m[ %(percent)3d%% ]\033[0m" %args
            self.write_message = "%(mode)s %(bar)s [ Top1: %(current_top1).2f Top5: %(current_top5).2f]  %(current)d/%(total)d [ %(percent)3d%% ]" %args
            print("\r" + message,file=self.output,end="")
        

    def done(self):
        self.current = self.total
        self()
        print("",file=self.output)
        with open(os.path.join(self.weights,self.model_name,self.Status ,self.current_time)+"/%s.txt"%self.model_name,"a") as f:
            print(self.write_message,file=f)
if __name__ == "__main__":

    from time import sleep
    progress = ProgressBar("Train",total_epoch=150,model_name="resnet159")
    for i in range(150):
        progress.total = 50
        progress.epoch = i
        progress.current_loss = 0.15
        progress.current_top1 = 0.45
        progress.current_top5= 0.65
        for x in range(50):
            progress.current = x
            progress()
            sleep(0.1)
        progress.done()
