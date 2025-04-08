#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:18:36 2021

@author: moreau
"""


from pycocotools.coco import COCO
from pycocotools.mask import decode
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import matplotlib.image as mpimg
import matplotlib.cm as cm
import datetime
import json
import copy
import cv2

def dc(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


with open('/path/to/test/annotation/annotation_test.json') as f:
  data = json.load(f)
  
with open('/path/to/model/output/on/test/set/coco_instances_results.json') as f:
  annotations = json.load(f)
  

for i in range(len(annotations)):
    annotations[i]['iscrowd']=0
    annotations[i]['id']=i

data['annotations']=annotations



# save predictions as annotations
with open('/path/to/save/inference/annotations/annotations.json', 'w') as output_json_file:
    json.dump(data, output_json_file)





image_directory ='/path/to/test/images/'
annotation_file = '/path/where/inference/is/saved/as/annotations/annotations.json'
mask_directory = '/path/to/test/references/'


example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['lesion'])
image_ids = example_coco.getImgIds(catIds=category_ids)

tirage = image_ids[np.random.randint(0, len(image_ids))]

possibles = []
for i in image_ids:
    if str(i)[:2]=="36":
        possibles.append(i)

for tirage in possibles:
    
    seg = []
    for item in data['annotations']:
        if item['image_id']==tirage:
            seg.append(item['segmentation'])
            

    image_data = example_coco.loadImgs(tirage)[0]
    
    image_data
    
    # load and display instance annotations
    image = io.imread(image_directory + image_data['file_name'])
    
    mask = io.imread(mask_directory + image_data['file_name'][:-3]+'png')
    print('image',tirage)
    for i in seg:
        print(dc(decode(i), mask[:,:,0]))
    print('')
        
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.imshow(image, cmap=cm.gray, vmin=0, vmax=255); plt.axis('off')
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    example_coco.showAnns(annotations, draw_bbox=False)
    

    # adapt according on what you want to see
    plt.title(tirage)
    ax1.imshow(image, cmap=cm.gray, vmin=0, vmax=255); plt.axis('off')
    ax1.imshow(mask, cmap=cm.gray, vmin=0, vmax=255, alpha=0.3); ax1.axis('off')
    plt.show()
    
    
    
    
