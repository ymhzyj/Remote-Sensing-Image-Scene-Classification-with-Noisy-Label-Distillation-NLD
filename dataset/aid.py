import os
import re
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import dataset.cls_transforms as tr
# import cls_transforms as tr
from torch.utils.data import Dataset, DataLoader


class AID_Noise_Train(Dataset):
    def __init__(self,
                 config,
                 root='data/AID/',
                 ):
        super().__init__()
        self._base_dir = root
        self.obj = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
                    'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond',
                    'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
        self.clean_images = []
        self.clean_targets = []
        self.noise_images = []
        self.noise_targets = []
        self.config = config
        self.train_states = config.train_status
        _clean_splits_file = os.path.join(
            self._base_dir, 'Split/', self.config.noise_type, str(self.config.percent), 'clean.txt')
        _noise_splits_file = os.path.join(
            self._base_dir, 'Split/', self.config.noise_type, str(self.config.percent), 'noise.txt')
        f = open(_clean_splits_file, 'r')
        clean_im_ids = f.readlines()
        f.close()
        f = open(_noise_splits_file, 'r')
        noise_im_ids = f.readlines()
        f.close()
        count = 0
        for i in clean_im_ids:
            cate = int(i.strip().split()[1])
            img_path = os.path.join(
                self._base_dir, "Images", self.obj[cate].strip(), str(i.strip().split()[0]))
            assert cate in range(0, 30)
            self.clean_images.append(img_path)
            self.clean_targets.append(cate)
        if self.train_states == 'Noise':
            self.clean_images.clear()
            self.clean_targets.clear()
        for i in noise_im_ids:
            cate = int(i.strip().split()[1])
            noise_obj_index=[s.lower() for s in self.obj].index(re.findall(r'[A-Za-z\']+', i)[0])
            img_path = os.path.join(
                self._base_dir, "Images",self.obj[noise_obj_index], str(i.strip().split()[0]))
            if self.obj[cate] != self.obj[noise_obj_index]:
                count += 1
            assert cate in range(0, 30)
            self.noise_images.append(img_path)
            self.noise_targets.append(cate)
        if self.train_states == 'Clean':
            self.noise_images.clear()
            self.noise_targets.clear()
            count = 0
        if self.train_states == 'Mix':
            self.noise_images = self.noise_images+self.clean_images
            self.noise_targets = self.noise_targets+self.clean_targets
            self.clean_images.clear()
            self.clean_targets.clear()
        self.clean_data = np.array(self.clean_images)
        self.clean_targets = np.array(self.clean_targets, dtype=np.int32)
        self.noise_data = np.array(self.noise_images)
        self.noise_targets = np.array(self.noise_targets, dtype=np.int32)
        self.noise_len = int(len(self.noise_data))
        self.clean_len = int(len(self.clean_data))
        if self.clean_len != 0 and self.noise_len != 0:
            min_len = int(min(len(self.clean_data), len(self.noise_data)))
            abs_len = abs(len(self.clean_data)-len(self.noise_data))
            randomindex_1 = [i for i in range(min_len)]
            randomindex_2 = np.random.randint(min_len, size=abs_len)
            self.randomindex = np.concatenate((randomindex_1, randomindex_2))
            np.random.shuffle(self.randomindex)
        print(
            f"Noise labels:{self.noise_len}   Clean labels:{self.clean_len}")
        print(f"Actually Noise Rate:{count/(self.noise_len+self.clean_len)}")

    def __len__(self):
        return max(len(self.clean_data), len(self.noise_data))

    def __getitem__(self, index):
        if (len(self.clean_data)) == 0:
            noise_img, noise_target = self.noise_data[index], self.noise_targets[index]
            noise_origin = self.noise_data.tolist()[index]
            noise_img = Image.open(noise_img).convert('RGB')
            noise_sample = {'image': noise_img,
                            'label': noise_target, 'origin': noise_origin}
            noise_sample = self.transform_img(noise_sample)
            return noise_sample
        if (len(self.noise_data)) == 0:
            clean_img, clean_target = self.clean_data[index], self.clean_targets[index]
            clean_origin = self.clean_data.tolist()[index]
            clean_img = Image.open(clean_img).convert('RGB')
            clean_sample = {'image': clean_img,
                            'label': clean_target, 'origin': clean_origin}
            clean_sample = self.transform_img(clean_sample)
            return clean_sample

        if (len(self.clean_data) > len(self.noise_data)):
            clean_img, clean_target = self.clean_data[index], self.clean_targets[index]
            clean_origin = self.clean_data.tolist()[index]
            clean_img = Image.open(clean_img).convert('RGB')
            clean_sample = {'image': clean_img,
                            'label': clean_target, 'origin': clean_origin}
            noise_img, noise_target = self.noise_data[self.randomindex[index]
                                                      ], self.noise_targets[self.randomindex[index]]
            noise_origin = self.noise_data.tolist()[self.randomindex[index]]
            noise_img = Image.open(noise_img).convert('RGB')
            noise_sample = {'image': noise_img,
                            'label': noise_target, 'origin': noise_origin}

        elif (len(self.clean_data) < len(self.noise_data)):
            clean_img, clean_target = self.clean_data[self.randomindex[index]
                                                      ], self.clean_targets[self.randomindex[index]]
            clean_origin = self.clean_data.tolist()[self.randomindex[index]]
            clean_img = Image.open(clean_img).convert('RGB')
            clean_sample = {'image': clean_img,
                            'label': clean_target, 'origin': clean_origin}
            noise_img, noise_target = self.noise_data[index], self.noise_targets[index]
            noise_origin = self.noise_data.tolist()[index]
            noise_img = Image.open(noise_img).convert('RGB')
            noise_sample = {'image': noise_img,
                            'label': noise_target, 'origin': noise_origin}
        else:
            clean_img, clean_target = self.clean_data[index], self.clean_targets[index]
            clean_origin = self.clean_data.tolist()[index]
            clean_img = Image.open(clean_img).convert('RGB')
            clean_sample = {'image': clean_img,
                            'label': clean_target, 'origin': clean_origin}
            noise_img, noise_target = self.noise_data[index], self.noise_targets[index]
            noise_origin = self.noise_data.tolist()[index]
            noise_img = Image.open(noise_img).convert('RGB')
            noise_sample = {'image': noise_img,
                            'label': noise_target, 'origin': noise_origin}
        clean_sample = self.transform_img(clean_sample)
        noise_sample = self.transform_img(noise_sample)
        return clean_sample, noise_sample

    def transform_img(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomFlip(),
            tr.RandomGaussianBlur(),
            tr.Resize(224),
            # tr.FixScaleCrop(600),
            tr.ToTensor(),
            tr.Normalize(mean=(0.324, 0.347, 0.307),
                         std=(0.197, 0.185, 0.185)),
        ])
        return composed_transforms(sample)


class AID_Clean(Dataset):
    def __init__(self,
                 root='data/AID/',
                 split='val',
                 ):
        super().__init__()
        self._base_dir = root
        self.obj = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
                    'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond',
                    'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
        self.images = []
        self.categories = []
        self.split = split

        _splits_file = os.path.join(
            self._base_dir, 'Split/', self.split+'_list.txt')
        f = open(_splits_file, 'r')
        im_ids = f.readlines()
        f.close()
        for i in im_ids:
            cate = int(i.strip().split()[1])
            img_path = os.path.join(
                self._base_dir, "Images", self.obj[cate].strip(), str(i.strip().split()[0]))
            assert cate in range(0, 30)
            self.images.append(img_path)
            self.categories.append(cate)
        self.images = np.array(self.images)
        self.categories = np.array(self.categories, dtype=np.int32)
        print(f"Images:{len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.categories[index]
        _origin = self.images.tolist()[index]
        sample = {'image': _img, 'label': _label, 'origin': _origin}
        if self.split == 'train':
            sample = self.transform_img(sample)
        else:
            sample = self.transform_test_img(sample)
        return sample

    def transform_img(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomFlip(),
            tr.RandomGaussianBlur(),
            tr.Resize(600),
            # tr.FixScaleCrop(600),
            tr.ToTensor(),
            tr.Normalize(mean=(0.324, 0.347, 0.307),
                         std=(0.197, 0.185, 0.185)),
        ])
        return composed_transforms(sample)

    def transform_test_img(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(600),
            tr.ToTensor(),
            tr.Normalize(mean=(0.324, 0.347, 0.307),
                         std=(0.197, 0.185, 0.185)),
        ])
        return composed_transforms(sample)


class AID_Semi(Dataset):
    def __init__(self,
                 root='data/AID/',
                 split='train',
                 percent=0
                 ):
        super().__init__()
        self._base_dir = root
        self.obj = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
                    'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond',
                    'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
        self.images = []
        self.categories = []
        self.split = split
        self.percent = percent
        if split == 'train':
            _splits_file = os.path.join(
                self._base_dir, 'Split/', 'Semi', '0', 'clean.txt')
            f = open(_splits_file, 'r')
            im_ids = f.readlines()
            f.close()
            for i in im_ids:
                cate = int(i.strip().split()[1])
                img_path = os.path.join(
                    self._base_dir, "Images", self.obj[cate].strip(), str(i.strip().split()[0]))
                assert cate in range(0, 30)
                self.images.append(img_path)
                self.categories.append(cate)
        elif split == 'plabel':
            im_ids = []
            for i in range(1, self.percent+1):
                _splits_file = os.path.join(
                    self._base_dir, 'Split/', 'Semi', 'Unlabeled', str(i)+'.txt')
                f = open(_splits_file, 'r')
                im_ids += f.readlines()
                f.close()
            for i in im_ids:
                cate = int(i.strip().split()[1])
                img_path = os.path.join(
                    self._base_dir, "Images", self.obj[cate].strip(), str(i.strip().split()[0]))
                assert cate in range(0, 30)
                self.images.append(img_path)
                self.categories.append(cate)
        self.images = np.array(self.images)
        self.categories = np.array(self.categories, dtype=np.int32)
        print(f"Images:{len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.categories[index]
        _origin = self.images.tolist()[index]
        sample = {'image': _img, 'label': _label, 'origin': _origin}
        if self.split == 'train':
            sample = self.transform_img(sample)
        else:
            sample = self.transform_test_img(sample)
        return sample

    def transform_img(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomFlip(),
            tr.RandomGaussianBlur(),
            tr.Resize(224),
            # tr.FixScaleCrop(600),
            tr.ToTensor(),
            tr.Normalize(mean=(0.324, 0.347, 0.307),
                         std=(0.197, 0.185, 0.185)),
        ])
        return composed_transforms(sample)

    def transform_test_img(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(224),
            tr.ToTensor(),
            tr.Normalize(mean=(0.324, 0.347, 0.307),
                         std=(0.197, 0.185, 0.185)),
        ])
        return composed_transforms(sample)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch AID Training')
    # Optimization options
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Miscs
    parser.add_argument('--manualSeed', type=int,
                        default=0, help='manual seed')
    # Device options
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # Method options
    parser.add_argument('--percent', type=float, default=0.8,
                        help='Percentage of noise')
    parser.add_argument('--begin', type=int, default=70,
                        help='When to begin updating labels')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Hyper parameter alpha of loss function')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Hyper parameter beta of loss function')
    parser.add_argument('--noise_type', type=str, default="Symm",
                        help='Asymmetric noise')
    parser.add_argument('--train_status', type=str, default="Double",
                        help='train_status')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Directory to output the result')
    parser.add_argument('--lamda', type=float, default=1000,
                        help='Hyper parameter beta of loss function')
    args = parser.parse_args()
    dst = AID_Noise_Train(config=args)
    val_dataloader = DataLoader(dst, batch_size=1)
    for i, (clean) in enumerate(val_dataloader):
        img, label, origin = clean['image'], clean['label'], clean['origin']
        # img = img.numpy()[:, ::-1, :, :]
        img = img[0].numpy().transpose((1, 2, 0))
        label = label[0].numpy()
        img *= (0.197, 0.185, 0.185)
        img += (0.324, 0.347, 0.307)
        img = img*255.0
        img = img.astype(np.uint8)
        # plt.imshow(img)
        # print(label)
        # print(origin)
        # plt.show()
