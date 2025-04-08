#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:57:30 2021

@author: moreau
"""

from pycocotools.mask import decode, merge
import json
from pycocotools.coco import COCO
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure, distance_transform_edt
import cv2
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


################fonction de calcul du DICE
def compute_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return 1
  inter = mask_gt & mask_pred
  volume_intersect = inter.sum()
  return 2*volume_intersect / volume_sum 



##########################fonctions de calcul de l'hausdorff distance
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 != np.count_nonzero(result) and 0 != np.count_nonzero(reference):
        # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
        # compute average surface distance
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
        sds = dt[result_border]
    
        return sds
    
    else:
        return 0
    


def hd(mask_pred, mask_gt, voxelspacing=None, connectivity=1):
    shd1 = __surface_distances(mask_pred, mask_gt, voxelspacing, connectivity)
    shd2 = __surface_distances(mask_gt, mask_pred, voxelspacing, connectivity)
    if type(shd1) == np.ndarray and type(shd2) == np.ndarray :
        hd1 = shd1.max()
        hd2 = shd2.max()
        hd = max(hd1, hd2)
        return hd




###########################fonction calcul volumetric similarity
def ravd(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    if 0 != vol2:
        return (vol1 - vol2) / float(vol2)




######################ouverture des fichiers


# initialize saving dictionnaries
res_dice = dict()
res_hd = dict()
res_raad = dict()


GT_dir = '/path/to/test/references/'

with open('/path/to/model/outputs/on/test/set/coco_instances_results.json') as f:
  donnees = json.load(f)

data_path=('/path/to/coco/annnotations/annotation_test.json')
coco_file=COCO(data_path)


category_ids = coco_file.getCatIds(catNms=['lesion'])
image_ids = coco_file.getImgIds(catIds=category_ids)


############ disctionnary with all slices
base_calcul = {'image_id' : [], 'prediction' : [], 'GT' : []}
dico_nombre_pred = {}

# iterate over test images
for item in donnees:
    
    prediction = decode(item['segmentation'])
    
    # if a lesion is predicted 
    if np.count_nonzero(prediction)!=0:
        image_data = coco_file.loadImgs(item['image_id'])[0]
        base_calcul['image_id'].append(image_data['file_name'][:-4])
        base_calcul['prediction'].append(prediction)
        annotation_ids = coco_file.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    
        GT = cv2.imread(os.path.join(GT_dir, image_data['file_name'][:-3]+'png'))
        GT = GT[:,:,0]>0
        GT = GT.astype(int)
        base_calcul['GT'].append(GT)
        
        # deal the several predictions for each image 
        if image_data['file_name'][:-4] not in dico_nombre_pred.keys() and np.count_nonzero(prediction)!=0:
            dico_nombre_pred[image_data['file_name'][:-4]] = 1
        elif image_data['file_name'][:-4] in dico_nombre_pred.keys() and np.count_nonzero(prediction)!=0:
            dico_nombre_pred[image_data['file_name'][:-4]]+=1

image_ids = coco_file.getImgIds(catIds=coco_file.getCatIds(catNms=['lesion']))          
cles = list(dico_nombre_pred.keys())
  

########################## calcul du DICE

for k in range(len(cles)):
    
    # when there is only one prediction for the images
    if dico_nombre_pred[cles[k]]==1:
        x=0
        i= base_calcul['image_id'][x]
        while i!=cles[k]:
            x+=1
            i= base_calcul['image_id'][x]
        Dice = compute_dice_coefficient(base_calcul['GT'][x], base_calcul['prediction'][x])
        res_dice[cles[k]] = res_dice.get(cles[k],[])+[Dice]

    
    else:
        positions=[]
        dsc=[]
        for x in range(len(base_calcul['image_id'])):
            if base_calcul['image_id'][x]==cles[k]:
                positions.append(x)

        # selection of the best mask over dice
        for pred in positions:
            dsc.append(compute_dice_coefficient(base_calcul['GT'][pred], base_calcul['prediction'][pred]))
        res_dice[cles[k]] = res_dice.get(cles[k],[])+[max(dsc)]


     
        
###########################calcul hausdorff distance

for k in range(len(cles)):
    
    # when there is only one prediction for the images
    if dico_nombre_pred[cles[k]]==1:
        x=0
        i= base_calcul['image_id'][x]
        while i!=cles[k]:
            x+=1
            i= base_calcul['image_id'][x]
        HD = hd(base_calcul['prediction'][x], base_calcul['GT'][x])
        if HD != None:
            res_hd[cles[k]] = res_hd.get(cles[k],[])+[HD]
    
    # otherwise, selection of the mask with the best dice
    else:
        positions=[]
        dsc=[]
        for x in range(len(base_calcul['image_id'])):
            if base_calcul['image_id'][x]==cles[k]:
                positions.append(x)
        P=positions[0]
        D= compute_dice_coefficient(base_calcul['GT'][positions[0]], base_calcul['prediction'][positions[0]])
        for pred in positions:
            if compute_dice_coefficient(base_calcul['GT'][pred], base_calcul['prediction'][pred])>D:
                D = compute_dice_coefficient(base_calcul['GT'][pred], base_calcul['prediction'][pred])
                P=pred
        HD = hd(base_calcul['prediction'][P], base_calcul['GT'][P])
        if HD != None:
            res_hd[cles[k]] = res_hd.get(cles[k],[])+[HD]


###########################calcul raad

for k in range(len(cles)):
    
    # when there is only one prediction for the images
    if dico_nombre_pred[cles[k]]==1:
        x=0
        i= base_calcul['image_id'][x]
        while i!=cles[k]:
            x+=1
            i= base_calcul['image_id'][x]
        RAAD = ravd(base_calcul['prediction'][x], base_calcul['GT'][x])
        if RAAD != None:
            res_raad[cles[k]] = res_raad.get(cles[k],[])+[RAAD]
    
    # otherwise, selection of the mask with the best dice
    else:
        positions=[]
        raad=[]
        for x in range(len(base_calcul['image_id'])):
            if base_calcul['image_id'][x]==cles[k]:
                positions.append(x)
        P=positions[0]
        D= compute_dice_coefficient(base_calcul['GT'][positions[0]], base_calcul['prediction'][positions[0]])
        for pred in positions:
            if compute_dice_coefficient(base_calcul['GT'][pred], base_calcul['prediction'][pred])>D:
                D = compute_dice_coefficient(base_calcul['GT'][pred], base_calcul['prediction'][pred])
                P=pred
        RAAD = ravd(base_calcul['prediction'][P], base_calcul['GT'][P])
        if RAAD != None:
            res_raad[cles[k]] = res_raad.get(cles[k],[])+[RAAD]                
            
moy_dice = dict()
moy_hd = dict()
moy_raad = dict()
for i in list(res_dice.keys()):
    moy_dice[i] = np.mean(res_dice[i])
    moy_hd[i] = np.mean(res_hd[i])
    moy_raad[i] = np.mean(res_raad[i])


print()
print(len(moy_dice)/len(os.listdir('/path/to/test/references/')))
print(np.mean(list(res_dice.values())), np.std(list(res_dice.values())))
print(np.mean(list(res_hd.values())), np.std(list(res_hd.values())))
print(np.mean(list(res_raad.values())), np.std(list(res_raad.values())))
print()



list_dice, list_hd, list_raad = [], [], []
for v in list(res_dice.values()):
    list_dice.append(v[0])
for v in list(res_hd.values()):
    list_hd.append(v[0])
for v in list(res_raad.values()):
    list_raad.append(v[0])



    
