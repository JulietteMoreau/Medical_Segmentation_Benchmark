#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:05:04 2021

@author: moreau
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:36:28 2021

@author: moreau
"""

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files



#informations globales

INFO = {
    "description": "Vérité terrain FLAIR",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
CATEGORIES = [
    {
        'id': 1,
        'name': 'lesion',
        'supercategory': 'shape',
    }
]




#initialisation of annotation


for etape in ['train', 'test', 'validation']:  #iterate over the three sets

    # prepare the output
    coco_output = {
            "info": INFO,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []}
    
    #counter to get the position in the path string of the patient number
    num_deb =1
    num_fin = num_deb + 4
    nom_deb = 0
    
    
    
    IMAGE_DIR = "/path/to/images/"+etape
    ANNOTATION_DIR = "/path/to/references/"+etape
    
    img_id = 1
    
    image_files = os.listdir(IMAGE_DIR)
    
        # go through each image
    for image_filename in image_files:
    
        # set an identity number for each image
        idtte = int(image_filename[num_deb:num_fin]+'0'+str(img_id))
        image = Image.open(os.path.join(IMAGE_DIR, image_filename))
        image_info = pycococreatortools.create_image_info(
            idtte, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
        
        annotation_filename = ANNOTATION_DIR+'/'+image_filename[nom_deb:-4]+".png"
        
        # only one class
        class_id = 1
        
        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
        binary_mask = np.asarray(Image.open(annotation_filename)
            .convert('1')).astype(np.uint8)
        
        # creation of the annotation
        annotation_info = pycococreatortools.create_annotation_info(
            idtte, idtte, category_info, binary_mask,
            image.size, tolerance=2)
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)
        
        img_id+=1
    
    # json saving
    with open('/path/to/save/annotations/annotation_'+etape+'.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    
