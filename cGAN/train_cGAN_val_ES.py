#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:44:24 2022

@author: moreau
"""

import argparse
import logging
import sys
from pathlib import Path
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from torch import Tensor
from generator_UNet import UNet
from discriminator import Discriminator
from torchvision.utils import save_image
import matplotlib.pyplot as plt

################################################################################################################dice_loss
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def limitcomma(value, limit=2):
  v = str(value).split(".")
  return float(v[0]+"."+v[1][:limit])


##############################################################################################################evaluate
def evaluate(generator, discriminator, dataloader, device, criterion_GAN, criterion_pixelwise):
    generator.eval()
    discriminator.eval()
    num_val_batches = len(dataloader)
    
    dice_score = 0
    L_G = 0
    L_D = 0
    nb = 0
    lambda_GAN = 1.  # Weights criterion_GAN in the generator loss
    lambda_pixel = 1.  # Weights criterion_pixelwise in the generator loss
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['cerveau'].type(Tensor).to(device), batch['GT'].type(Tensor).to(device)
        mask_true = mask_true>0
        valid = Tensor(np.ones((mask_true.size(0), 1, 3, 3))).to(device)
        fake = Tensor(np.zeros((mask_true.size(0), 1, 3, 3))).to(device)
        nb += image.shape[0]
            
        with torch.no_grad():
            # predict the mask
            mask_pred = generator(image.float())
            pred_fake = discriminator(torch.argmax(mask_pred, dim=1).reshape((mask_pred.shape[0], 1,192,192)), image)   
            pred_real = discriminator(mask_true, image)
            
            #loss generator
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(mask_pred, mask_true.long().squeeze(1)) + dice_loss(F.softmax(mask_pred, dim=1).float(), F.one_hot(mask_true.long().squeeze(1), generator.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)
            loss_generator = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel
            
            #loss discriminator
            loss_real = criterion_GAN(pred_real, valid)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_discriminator = 0.5 * (loss_real + loss_fake)

            L_G += loss_generator.item() #* image.size(0)
            L_D += loss_discriminator.item()

    generator.train()
    discriminator.train()
    
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return L_G/nb, L_D/nb
    
    return L_G/nb, L_D/nb



################################################################################################################train
def train_net(train_loader, train_ds, val_loader, outdir, patience=5, num_epoch=500,
               lr=0.0001, beta1=0.9, beta2=0.999, amp=False):

    # initialization of loss savings for generator and discriminator
    loss_values_D=[]
    loss_values_G=[]    
    loss_val_D = []
    loss_val_G = []
    
    # initialization of early stopping
    trigger_times = 0
    last_loss = 100.0000
    
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using cuda device: {cuda}")  # check if GPU is used

    # Tensor type (put everything on GPU if possible)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Output folder
    if not os.path.exists(outdir+"/images"):
        os.makedirs(outdir+"/images")
        
    if not os.path.exists(outdir+"/checkpoints"):
        os.makedirs(outdir+"/checkpoints")  


    # Loss functions
    criterion_GAN = torch.nn.BCEWithLogitsLoss()  # A loss adapted to binary classification
    criterion_pixelwise = torch.nn.CrossEntropyLoss()  # A loss for a pixel-wise comparison of images

    lambda_GAN = 1.  # Weights criterion_GAN in the generator loss
    lambda_pixel = 1.  # Weights criterion_pixelwise in the generator loss

    # Initialize generator and discriminator
    generator = UNet()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    # Optimizers
    optimizer_generator = torch.optim.RMSprop(generator.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    
    
    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(num_epoch):
        
        running_loss_G = 0.0
        running_loss_D = 0.0
        
        for i, batch in enumerate(train_loader):

            # Inputs CT and Mask
            real_CT = batch["cerveau"].type(Tensor)
            real_Mask = batch["GT"].type(Tensor)
            real_Mask = real_Mask>0

            # Create labels
            valid = Tensor(np.ones((real_Mask.size(0), 1, 3, 3)))
            fake = Tensor(np.zeros((real_Mask.size(0), 1, 3, 3)))

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_generator.zero_grad()

            # GAN loss
            fake_Mask = generator(real_CT.float())
            pred_fake = discriminator(torch.argmax(fake_Mask, dim=1).reshape((fake_Mask.shape[0], 1,192,192)), real_CT)   
            loss_GAN = criterion_GAN(pred_fake, valid)

            # segmentation loss
            loss_pixel = criterion_pixelwise(fake_Mask, real_Mask.long().squeeze(1)) + dice_loss(F.softmax(fake_Mask, dim=1).float(), F.one_hot(real_Mask.long().squeeze(1), generator.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)

            # Total loss
            loss_generator = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel

            optimizer_generator.zero_grad(set_to_none=True)
            grad_scaler.scale(loss_generator).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
            grad_scaler.step(optimizer_generator)
            grad_scaler.update()


            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Real loss
            pred_real = discriminator(real_Mask, real_CT)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            fake_Mask = generator(real_CT.float())
            pred_fake = discriminator(torch.argmax(fake_Mask, dim=1).reshape((fake_Mask.shape[0], 1,192,192)), real_CT)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_discriminator = 0.5 * (loss_real + loss_fake)

            # Compute the gradient and perform one optimization step
            loss_discriminator.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            optimizer_discriminator.step()
                

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = num_epoch * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
                "[G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch + 1,
                    num_epoch,
                    i,
                    len(train_loader),
                    loss_discriminator.item(),
                    loss_generator.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )
            running_loss_G += loss_generator.item() * real_CT.size(0) 
            running_loss_D += loss_discriminator.item() * real_CT.size(0) 
            
            
        # --------------
        #  Evaluation
        # --------------
            
        val_loss_G, val_loss_D = evaluate(generator, discriminator, val_loader, device, criterion_GAN, criterion_pixelwise)
        optimizer_generator.zero_grad(set_to_none=True)
        optimizer_discriminator.zero_grad(set_to_none=True)

        sys.stdout.write(
            "\rValidation Dice score %f,%f"
            % (val_loss_G, val_loss_D))
        
        loss_val_G.append(val_loss_G)
        loss_val_D.append(val_loss_D)
        

        # --------------
        #  early stopping
        # --------------
        
        # check if the validation loss stops improving
        if limitcomma(val_loss_G, 4) > limitcomma(last_loss,4):
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            # check if the patience is over
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                
                # saving checkpoints
                torch.save(net.state_dict(), outdir+"/checkpoints/checkpoint.pth")
                logging.info(f'Checkpoint {epoch + 1} saved!')
                
                # plt loss
                plt.rcParams["figure.figsize"] = (16,12)
                plt.plot(loss_values_D, color='darkred')
                plt.plot(loss_values_G, color='firebrick')
                plt.plot(loss_val_D, color='royalblue', linestyle='--')
                plt.plot(loss_val_G, color='navy', linestyle='--')
                plt.legend(['entrainement discriminateur', 'entrainement générateur', 'validation discriminateur', 'validation générateur'])
                plt.title("fonction de coût")
                plt.xlabel('Epoques')
                plt.savefig(outdir+"/images/loss.png")

                return net
            
        else:
            trigger_times = 0

        last_loss = val_loss_G


        # --------------
        #  end of training
        # --------------
        
        if epoch==num_epoch-1:
            torch.save(net.state_dict(), outdir+"/checkpoints/checkpoint_epoch{}.pth".format(epoch + 1))
            logging.info(f'Checkpoint {epoch + 1} saved!')
            

    plt.rcParams["figure.figsize"] = (16,12)
    plt.plot(loss_values_D, color='darkred')
    plt.plot(loss_values_G, color='firebrick')
    plt.plot(loss_val_D, color='royalblue', linestyle='--')
    plt.plot(loss_val_G, color='navy', linestyle='--')
    plt.legend(['entrainement discriminateur', 'entrainement générateur', 'validation discriminateur', 'validation générateur'])
    plt.title("fonction de coût")
    plt.xlabel('Epoques')
    plt.savefig(outdir+"/images/loss.png")


    return net

