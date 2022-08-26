import glob
import logging
from multiprocessing.resource_sharer import stop
import os
import cv2
import random
import torch
import numpy as np
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset
import pandas as pd  



## function to create the labels
def get_label(batch_size):
    "We choose '0' for the fake labels and '1' for the real ones"
    return torch.ones(batch_size, 1)*0.9,torch.zeros(batch_size, 1)




# define our data augmentation transform
def get_transform():
    transform = A.Compose([
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
    A.Normalize(),
    ])
    return transform

class DataloaderDCGAN(Dataset):
    def __init__(self, indir,img_size):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        random.shuffle(self.in_files)
        self.transform = get_transform()
        self.img_size = img_size

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.resize(img, (self.img_size,self.img_size), interpolation= cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        return img

## debug
if __name__ =='__main__': 
    data = DataloaderDCGAN(indir='./train/',kind = 'train')