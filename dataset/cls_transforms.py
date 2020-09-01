import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageFilter,ImageEnhance
from random import randint
import skimage
import warnings
import math
import numbers
class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, sample):
        old_image = sample['image']
        label = sample['label']
        origin=sample['origin']
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return {'image': new_image,
                'label': label,
                'origin':origin}
# class RandomHorizontalFlip(object):
#     def __call__(self,sample):
#         img = sample['image']
#         label = sample['label']
#         origin=sample['origin']
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#         return {'image': img,
#                 'label': label,
#                 'origin':origin}
class RandomFlip(object):
    def __call__(self,sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img,
                'label': label,
                'origin':origin}

class RandomGaussianBlur(object):
    def __call__(self,sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': label,
                'origin':origin}
                
class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        elif w<h:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        else:
            ow=self.crop_size
            oh=self.crop_size
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': label,
                'origin':origin}


class RandomZoom(object):
    def __init__(self,zoom_range):
        self.zoom_range = zoom_range

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        # w,h=img.size
        scale=random.uniform(self.zoom_range[0],self.zoom_range[1])
        # scale=random.randint(256, 480)
        # w, h = img.size

        
        ow=int(scale)
        oh=int(scale)
        
        img = img.resize((ow, oh), Image.BILINEAR)
        
        # if label==0:
        #     cropw=random.randint(20,140)
        #     croph=random.randint(20,140)
            
        #     crop_x=random.randint(0,img.size[0]-cropw-1)
        #     crop_y=random.randint(0,img.size[1]-croph-1)
            
        #     img=img.crop((crop_x,crop_y,crop_x+cropw,crop_y+croph))
        
        return {'image': img,
                'label': label,
                'origin':origin}

class Resize(object):
    def __init__(self, resize):
        self.resize = resize
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        img=img.resize((self.resize,self.resize))
        return {'image': img,
                'label': label,
                'origin':origin}

class RandomCrop(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
#        scale=[0.75,1.0,1.25]
#        tmp=random.randint(0,2)
        
        w, h = img.size
        size=224
        new_left=randint(0,w-size)
        new_upper=randint(0,h-size)
        img=img.crop((new_left,new_upper,size+new_left,size+new_upper))
        return {'image': img,
                'label': label,
                'origin':origin}

class MiddleCrop(object):
    def __init__(self,size):
        self.size = size
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
#        scale=[0.75,1.0,1.25]
#        tmp=random.randint(0,2)
        
        w, h = img.size
        new_left=(w-self.size)/2
        new_upper=(h-self.size)/2
        img=img.crop((new_left,new_upper,self.size+new_left,self.size+new_upper))
        return {'image': img,
                'label': label,
                'origin':origin}

class RandomRotation(object):
    def __init__(self,angles=360):
        self.angles = angles
    def __call__(self,sample):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        random_angle = np.random.randint(0, self.angles)
        img=img.rotate(random_angle, Image.BICUBIC)
        return {'image': img,
                'label': label,
                'origin':origin}

class randomColor(object):
    def __call__(self,sample):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        img=ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
        return {'image': img,
                'label': label,
                'origin':origin}

class randomBrightness(object):
    def __init__(self, random_factors):
        self.random_factors = random_factors
    def __call__(self,sample):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        color_image = img
        random_factor = np.random.uniform(self.random_factors[0], self.random_factors[1])  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        img=brightness_image
        return {'image': img,
                'label': label,
                'origin':origin}

class randomGaussian(object):

    def __init__(self,mean=0.2,sigma=0.3):
        self.mean = mean
        self.sigma = sigma
    def __call__(self, sample):

        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        img=np.asarray(img)
        img.flags.writeable = True
        img=skimage.util.random_noise(img)
        # width, height = img.shape[:2]
        # img_r = self.GaussianNoise(img[:, :, 0].flatten())
        # img_g = self.GaussianNoise(img[:, :, 1].flatten())
        # img_b = self.GaussianNoise(img[:, :, 2].flatten())
        # img[:, :, 0] = img_r.reshape([width, height])
        # img[:, :, 1] = img_g.reshape([width, height])
        # img[:, :, 2] = img_b.reshape([width, height])
        img=Image.fromarray(np.uint8(img))
        return {'image': img,
                'label': label,
                'origin':origin}
    # def GaussianNoise(self,im):
    #     for _i in range(len(im)):
    #         im[_i] += random.gauss(self.mean, self.sigma)
    #     return im
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), inplace=False):
        # self.mean = torch.from_numpy(np.array(mean).astype(np.float32)).float()
        # self.std = torch.from_numpy(np.array(std).astype(np.float32)).float()
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        img = F.normalize(img, self.mean, self.std, self.inplace)
        # img /= 255.0
        # img -= self.mean
        # img /= self.std

        return {'image': img,
                'label': label,
                'origin':origin}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        label = sample['label']
        origin=sample['origin']
        # img=F.to_tensor(img)
        # label=F.to_tensor(label)
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        label = np.array(label).astype(np.float32)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()

        return {'image': img,
                'label': label,
                'origin':origin}