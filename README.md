本代码来自于论文"Remote Sensing Image Scene Classification with Noisy Label Distillation"  
发表于Remote Sensing  

目录结构:  
根目录/   
        config/ 训练参数配置文件，分别针对不同的数据集/  
                AID  
                NWPU45  
                UCMerced  
        data/ 数据集存放位置，以及数据预处理文件  
                NWPU-RESISC45  
                                                Images/类别  
                                                Split/训练测试划分  
                                                NWPU45_split.py  
                UCMerced_LandUse  
                                                Images/类别  
                                                Split  
                                                UCMerced_split.py  
        dataset/ 数据集预处理文件   
                aid  
                nwpu45    
                ucmerced 数据集预处理    
                cls_transforms 通用预处理函数    
        losses/ 其他Pytorch实现的损失函数   
        models/ 实验模型    
                getnet 用于主函数获取模型的端口    
                Resnet    
                VGG 模型实现    
        runs/ 用于记录实验结果   
        utils/    
                progress_bar 进度条显示    
                utils 精度测量，模型保存函数等   
        main_数据集名称_noisy 噪声标签实验    
        main_数据集名称_clean 干净标签实验    

配置方法   
        提出的方法：   
                dataset_split.py用于分割数据集，并且生成固定的噪声标签和干净标签，方便比较   
                main_dataset_noisy.py   
                        train_status可固定为Double，Finetune=False   
                        可以使用的噪声有三种  
                        其中Asym，Symm对应了percent分别是0.1-0.4和0.2，0.4，0.6，0.8   
                        Semi不能直接使用，需要生成伪标签后才可以使用，percent对应1-5  
                        此外可以直接使用无噪声完整数据集训练，对应None的noise_type，percent=0   
                main_dataset_single.py  
                        用于训练基准网络，train_status可使用除double外的所有形式   
                        train_status=Clean,Finetune=False表示完全干净数据   
                                一般基准训练，noise_type可以设置为Asym，Symm和None，其中None的percent只能为0,其余可以设置，得到包含相应比例的噪声数据集中的干净数据   
                                生成伪标签，noise_type设置为Semi，percent=0,可以利用1/6的数据对训练网络   
                        train_status=Noise,Finetune=False完全噪声数据集，Mix混合数据的数据集   
                                可以设置对应的Asym和Symm/暂时没有Semi的，percent分别是0.1-0.4和0.2，0.4，0.6，0.8   
                        train_status=Clean_FT,表示用Asym或者Symm类数据集，噪声先对网络进行训练，然后利用干净数据微调，第一次训练阶段需要Finetune=False，第二次训练Finetune=True
                        train_status=Noise_FT同上   
                make_dataset_plabel.py    
                        用于生成伪标签，分别将用过single文件训练得到的两个在Semi配置下的网络的time和time_2以及model和model_2,noise_type设置为Semi，percent=0   



论文引用格式：  

@article{zhang2020remote,   
  title={Remote Sensing Image Scene Classification with Noisy Label Distillation},   
  author={Zhang, Rui and Chen, Zhenghao and Zhang, Sanxing and Song, Fei and Zhang, Gang and Zhou, Quancheng and Lei, Tao},   
  journal={Remote Sensing},   
  volume={12},   
  number={15},   
  pages={2376},   
  year={2020},   
  publisher={Multidisciplinary Digital Publishing Institute}   
}   
