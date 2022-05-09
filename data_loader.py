# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random
import os
import cv2
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision.transforms.functional as tf
import torchvision.transforms as T
import pdb
from alive_progress import alive_bar
IMG_MAX = 32722
IMG_MIN = -31822
def normalize(img_depth):
    img_depth = img_depth.float()
    img_depth = (img_depth - IMG_MIN) / (IMG_MAX - IMG_MIN)
    return img_depth

class RecClothDataset(Dataset):

    def __init__(self, root_dir,num_masks, use_transform=True, datasize=None):
        self.root_dir = root_dir
        self.use_transform = use_transform
        self.num_masks = num_masks

        self.rgb_root = os.path.join(self.root_dir,"rgb")
        self.depth_root = os.path.join(self.root_dir,"depth")
        self.masks_root = os.path.join(self.root_dir,"masks")
        ##Following one indexing
        masks = os.listdir(self.masks_root)
        self.masks_roots = [os.path.join(self.masks_root,f) for f in masks]
        self.img_paths = os.listdir(self.rgb_root)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        imidx = self.img_paths[idx].replace(".npy", "")
        img_path = os.path.join(self.rgb_root, imidx+".npy")
        depth_path = os.path.join(self.depth_root, imidx+".npy")

        depth_npy = np.load(depth_path)
        depth_npy[np.isnan(depth_npy)] = max_d = np.nanmax(depth_npy)
        img_npy = np.load(img_path)
        img_depth = Image.fromarray(depth_npy)
        img_rgb = Image.fromarray(img_npy,mode="RGB")



        transform = T.Compose([T.ToTensor()])
        labels = []
        for m in reversed(range(self.num_masks)):
            m_npy = np.load(os.path.join(self.masks_roots[m],imidx+".npy"))
            img_hsv = Image.fromarray(m_npy)
            labels.append(img_hsv)
        
        if self.use_transform:
            if random.random() > 0.5:
                img_rgb = tf.hflip(img_rgb)
                img_depth = tf.hflip(img_depth)
                labels = [tf.hflip(l) for l in labels]
            if random.random() > 0.5:
                img_rgb = tf.vflip(img_rgb)
                img_depth = tf.vflip(img_depth)
                labels = [tf.vflip(l) for l in labels]
            if random.random() > 0.9:
                angle = T.RandomRotation.get_params([-30, 30])
                img_rgb = tf.rotate(img_rgb, angle, resample=Image.NEAREST)
                img_depth = tf.rotate(img_depth, angle, resample=Image.NEAREST)
                labels = [tf.rotate(l,angle,resample=Image.NEAREST) for l in labels]
        img_rgb = transform(img_rgb)
        img_depth = transform(img_depth)

        labels = [transform(l) for l in labels]
        
        label = torch.cat(labels, 0)
        img_depth = normalize(img_depth)

        sample = {'rgb': img_rgb, 'X': img_depth, 'Y': label}

        return sample
    
    def get_max_min(self):
        self.use_transform = False
        maxi = -100000
        mini = 100000
        with alive_bar(len(self)) as bar:
            for i in range(len(self)):
                data = self[i]
                inp = data['X'][0,:,:].numpy()
                cur_max = inp.max()
                cur_min = inp.min()
                print(cur_max, cur_min)
                if(cur_max > maxi):
                    maxi = cur_max
                if(cur_min < mini):
                    mini = cur_min
                bar()
        print(maxi, mini)


if __name__ == "__main__":
    train_data = RecClothDataset(root_dir="/media/YertleDrive4/layer_grasp/dataset/2cloth_rec/val", num_masks=2, use_transform=True)
    num= 1
    # train_data.get_max_min()
    for i in range(num):
        sample = train_data[i]
        print(i, sample['X'].size())
        print(sample['X'].max(), sample['X'].min(), sample['X'].type())
        print(sample['Y'].max(), sample['Y'].min(), sample['Y'].type())
        print(sample['rgb'].max(), sample['rgb'].min(), sample['rgb'].type())

        a = sample['Y'].numpy()
        print(sample['X'].numpy().shape)
        print(sample['rgb'].numpy().shape)
        print(sample['Y'].numpy().shape)


