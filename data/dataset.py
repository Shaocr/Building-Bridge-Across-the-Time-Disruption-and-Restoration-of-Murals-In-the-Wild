import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import random

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(cond_dir, dir=None):
    if os.path.isfile(cond_dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        cond_imgs = []
        assert os.path.isdir(cond_dir), '%s is not a valid directory' % cond_dir
        for root, _, fnames in sorted(os.walk(cond_dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    cond_imgs.append(path)
                    if dir is not None:
                        images.append(os.path.join(dir, fname))
    return images, cond_imgs

def pil_loader(path):
    return Image.open(path).convert('RGB')

class DunHuangDataset(data.Dataset):
    def __init__(self, cond_data_root, data_root=None, data_len=-1, image_size=[256, 256], loader=pil_loader):
        self.test = data_root is None
        imgs, cond_imgs = make_dataset(cond_data_root, data_root)
        if data_len > 0:
            self.imgs, self.cond_imgs = imgs[:int(data_len)], cond_imgs[:int(data_len)]
        else:
            self.imgs = imgs
            self.cond_imgs = cond_imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        
        cond_path = self.cond_imgs[index]
        cond_img = self.tfs(self.loader(cond_path))
        
        img = 0
        attention_img = 0
        if not self.test:
            path = self.imgs[index]
            img = self.tfs(self.loader(path))
            attention_img = torch.sigmoid(torch.mean(torch.square(img-cond_img), axis=0).unsqueeze(0))

        ret['gt_image'] = img
        ret['cond_image'] = cond_img
        ret['attention_image'] = attention_img
        ret['path'] = cond_path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.cond_imgs)