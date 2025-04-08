#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:25:22 2022

@author: moreau
"""

# torch stuff
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torchsummary and torchvision
from torchsummary import summary
from torchvision.utils import save_image

# Common python packages
import os
import glob
import sys

#monai stuff
from monai.transforms import RandSpatialCropSamplesD,SqueezeDimd, SplitChannelD,RandWeightedCropd,\
    LoadImageD, EnsureChannelFirstD, AddChannelD, ScaleIntensityD, ToTensorD, Compose, CropForegroundd,\
    AsDiscreteD, SpacingD, OrientationD, ResizeD, RandAffineD, CopyItemsd, OneOf, RandCoarseDropoutd, RandFlipd
from monai.data import CacheDataset

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#################################################################################################################################load data
#recovery of data and output files
dossier = sys.argv[1]
outdir = sys.argv[3]


if not os.path.exists(outdir):
    os.makedirs(outdir)   

KEYS = ("cerveau", "GT")

#recovery of all data separated into three sets
train_dir = dossier + '/image/train/'
train_dir_label = dossier + '/ref/train/'
val_dir = dossier + '/image/validation/'
val_dir_label = dossier + '/ref/validation/'
test_dir = dossier + '/image/test/'
test_dir_label = dossier + '/ref/test/'


train_images = sorted(glob.glob(train_dir + "*.jpg"))
train_labels = sorted(glob.glob(train_dir_label + "*.png"))

val_images = sorted(glob.glob(val_dir + "*.jpg"))
val_labels = sorted(glob.glob(val_dir_label + "*.png"))

test_images = sorted(glob.glob(test_dir + "*.jpg"))
test_labels = sorted(glob.glob(test_dir_label + "*.png"))

#creation of lists for the various data sets
train_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images, train_labels)
]
val_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images, val_labels)
]
test_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(test_images, test_labels)
]
print("Number Train files: "+str(len(train_files)))
print("Number val files: "+str(len(val_files)))
print("Number test files: "+str(len(test_files)))

# Create dataloaders
xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])


bs = 12 #batch size, can be modified
train_ds = CacheDataset(data=train_files, transform=xform, num_workers=10)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=xform, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True)
test_ds = CacheDataset(data=test_files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


#modification of channels to be adapted to BW images
for i, batch in enumerate(train_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()
for i, batch in enumerate(val_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()
for i, batch in enumerate(test_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()
      
    
################################################################################################################learning parameters
# recovery of learning rate
lr = sys.argv[2]


# Number of epochs and patience for early-stopping, can be modified
num_epoch = 100
patience = 5



# ################################################################################################################generator Unet
from generator_UNet import UNet
# Summary of the generator
summary(UNet().cuda(), (1, 192, 192))

#######################################################################################################################load training
from train_generator_val_ES import train_net
generator = train_net(train_loader, train_ds, val_loader, outdir, patience = patience, num_epoch=num_epoch, lr=lr)


