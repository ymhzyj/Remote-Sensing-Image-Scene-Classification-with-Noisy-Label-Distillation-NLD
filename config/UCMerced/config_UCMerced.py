class DefaultConfigs(object):
    # 1.string parameters
    dataset = 'UCMerced'#dataset name
    dataroot='data/UCMerced_LandUse'
    num_classes = 21
    epochs= 200
    weights= "./runs/"+dataset+"/"
    gpus= "0"


    # 2.numeric parameters
    workers = 0
    batch_size= 32
    noisebatch_size = 16
    test_batch_size = 1
    patience = 80 #
    train_ratio=0.9
    seed= 888
    lr=1e-4
    # lr = 5e-4
    # weights_lr = 1e-3
    # clean_lr= 0.01
    # noise_lr = 0.1
    # lr_decay= 5e-4
    weight_decay= 1e-2



    #需要修改
    percent=0.8 #0.1-0.9 # 1,2,3,4,5 for Semi 
    time = "2020-04-29-18-13-04"#测试时找对应的文件夹 fine-tune也需要
    model_name = "Resnet34" # Dnet50-34,Dnet34-34,ResNet50,VGG16,Resnet34,VGG16,Resnet50-VGG16
    noise_type="Asym" # Asym,Symm,Semi,None
    train_status="Mix" #Clean,Noise,Mix,Double,Clean_FT(clean fine-tuning先noise,后clean,只需要更改Finetune和时间，先True后False),Mix_FT,Mix_SCE
    Finetune = False #第一次训练False，第二次微调True
    Status= noise_type+'/'+str(percent)+'/'+train_status
    #Plabels
    time_2="2020-04-23-20-03-51"#only for Pseudo-Labelling
    model_name_2="ResNet50" #only for Pseudo-Labelling
config_cifar10= DefaultConfigs()
