import os
from random import shuffle
import numpy as np
import shutil


def text_save(filename, data, class_names):  # filename为写入文件的路径，data为要写入数据列表.
    if isinstance(class_names, int):
        class_names = np.full(len(data), class_names)
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i])
        s = s.replace("'", '')+" "+str(class_names[i])+'\n'
        # s = s.replace("'",'')+' '+str(class_name)+'\n'  #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def addSymmNoise(data, class_name):
    imglist = np.array(data)
    truelabels = np.full(len(imglist), class_name)
    noise_precent = [0.2, 0.4, 0.6, 0.8]
    for p in noise_precent:
        if not os.path.exists('data/UCMerced_LandUse/Split/Symm/'+str(p)):
            os.makedirs('data/UCMerced_LandUse/Split/Symm/'+str(p))
        categories = truelabels.copy()
        noisy_len = int(len(imglist)*p)
        noisy_data = np.zeros(noisy_len, dtype=np.uint8)
        noisy_targets = np.zeros(noisy_len, dtype=np.int32)
        clean_len = len(imglist)-noisy_len
        clean_data = np.zeros(clean_len, dtype=np.uint8)
        clean_targets = np.zeros(clean_len, dtype=np.int32)
        indices = np.random.permutation(len(imglist))
        for i, idx in enumerate(indices):
            if i < noisy_len:
                categories[idx] = np.random.randint(21, dtype=np.int32)
        noisy_data = imglist[indices[0:noisy_len]]
        noisy_targets = categories[indices[0:noisy_len]]
        clean_data = imglist[indices[noisy_len:]]
        clean_targets = categories[indices[noisy_len:]]
        assert len(noisy_data)+len(clean_data) == len(imglist)
        text_save(os.path.join('data/UCMerced_LandUse/Split/Symm',
                               str(p), 'clean.txt'), clean_data, clean_targets)
        text_save(os.path.join('data/UCMerced_LandUse/Split/Symm',
                               str(p), 'noise.txt'), noisy_data, noisy_targets)


def addAsymNoise(data, class_name):
    imglist = np.array(data)
    truelabels = np.full(len(imglist), class_name)
    noise_precent = [0.2, 0.4, 0.6, 0.8]
    for p in noise_precent:
        if not os.path.exists('data/UCMerced_LandUse/Split/Asym/'+str(p)):
            os.makedirs('data/UCMerced_LandUse/Split/Asym/'+str(p))
        categories = truelabels.copy()
        noisy_len = int(len(imglist)*p)
        noisy_data = np.zeros(noisy_len, dtype=np.uint8)
        noisy_targets = np.zeros(noisy_len, dtype=np.int32)
        clean_len = len(imglist)-noisy_len
        clean_data = np.zeros(clean_len, dtype=np.uint8)
        clean_targets = np.zeros(clean_len, dtype=np.int32)
        noisy_indices = np.zeros(noisy_len, dtype=np.int32)
        clean_indices = np.zeros(clean_len, dtype=np.int32)
        indices = np.random.permutation(len(imglist))
        indices_len = int(p * len(indices))
        noisy_indices = indices[0:indices_len]
        clean_indices = indices[indices_len:]
        if class_name == 5:
            noisy_targets = np.zeros(noisy_len, dtype=np.int32)
        elif class_name == 17:
            noisy_targets = np.full(noisy_len, 1)
        elif class_name == 1:
            noisy_targets = np.full(noisy_len, 17)
        elif class_name == 20:
            noisy_targets = np.full(noisy_len, 2)
        elif class_name == 16:
            noisy_targets = np.full(noisy_len, 3)
        elif class_name == 13:
            noisy_targets = np.full(noisy_len, 10)
        elif class_name == 8:
            noisy_targets = np.full(noisy_len, 14)
        elif class_name == 14:
            noisy_targets = np.full(noisy_len, 8)
        elif class_name == 19:
            noisy_targets = np.full(noisy_len, 4)
        elif class_name == 15:
            noisy_targets = np.full(noisy_len, 13)
        elif class_name == 12:
            noisy_targets = np.full(noisy_len, 6)
        elif class_name == 6:
            noisy_targets = np.full(noisy_len, 12)
        else:
            noisy_targets = categories[noisy_indices]
        noisy_data = imglist[noisy_indices]
        clean_data = imglist[clean_indices]
        clean_targets = categories[clean_indices]
        assert len(noisy_data)+len(clean_data) == len(imglist)
        text_save(os.path.join('data/UCMerced_LandUse/Split/Asym',
                               str(p), 'clean.txt'), clean_data, clean_targets)
        text_save(os.path.join('data/UCMerced_LandUse/Split/Asym',
                               str(p), 'noise.txt'), noisy_data, noisy_targets)
def  splitSemi(data,class_name):
        if not os.path.exists('data/UCMerced_LandUse/Split/Semi/'):
            os.makedirs('data/UCMerced_LandUse/Split/Semi/Unlabeled')
        imglist = np.array(data)
        truelabels = np.full(len(imglist), class_name)
        img=np.split(imglist,6)
        labels=np.split(truelabels,6)
        for i in range(0,6):
            text_save(os.path.join('data/UCMerced_LandUse/Split/Semi/Unlabeled',str(i)+'.txt'),img[i],labels[i])
            if not os.path.exists(os.path.join('data/UCMerced_LandUse/Split/Semi/',str(i))):
                os.makedirs(os.path.join('data/UCMerced_LandUse/Split/Semi/',str(i)))
            text_save(os.path.join('data/UCMerced_LandUse/Split/Semi/'+str(i),'clean.txt'),img[0],labels[0])
            



def main():
    class_names = [
        "agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential", "forest", "freeway", "golfcourse", "harbor",
        "intersection", "mediumresidential", "mobilehomepark", "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"
    ]
    root = 'data/UCMerced_LandUse/Images'
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    assert train_ratio+val_ratio+test_ratio == 1
    if os.path.exists('data/UCMerced_LandUse/Split'):
        shutil.rmtree('data/UCMerced_LandUse/Split')
    os.makedirs('data/UCMerced_LandUse/Split')
    os.makedirs('data/UCMerced_LandUse/Split/Symm')
    os.makedirs('data/UCMerced_LandUse/Split/Asym')
    # f= open('data/UCMerced_LandUse/Split/train_list.txt')
    # im_ids=f.readlines()
    # f.close()
    # train_list=[]
    # for i in range(21):
    #     train_list=[x.strip().split(' ')[0]  for x in im_ids[i*60:(i+1)*60]]
    #     class_names=i
    #     addAsymNoise(train_list, class_names)
    for i in range(len(class_names)):
        imglist = os.listdir(os.path.join(root, class_names[i]))
        shuffle(imglist)
        lengths = len(imglist)
        train_len = int(lengths*train_ratio)
        val_len = int(lengths*val_ratio)
        test_len = int(lengths-train_len-val_len)
        trainlist = imglist[0:train_len]
        vallist = imglist[train_len:train_len+val_len]
        testlist = imglist[train_len+val_len:train_len+val_len+test_len]
        assert(trainlist+vallist+testlist == imglist)
        text_save('data/UCMerced_LandUse/Split/train_list.txt', trainlist, i)
        addSymmNoise(trainlist, i)
        addAsymNoise(trainlist, i)
        splitSemi(trainlist,i)
        text_save('data/UCMerced_LandUse/Split/val_list.txt', vallist, i)
        text_save('data/UCMerced_LandUse/Split/test_list.txt', testlist, i)


if __name__ == "__main__":
    main()
