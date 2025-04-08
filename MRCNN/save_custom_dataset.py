#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:01:42 2024

@author: moreau
"""

from detectron2.data.datasets import register_coco_instances

from detectron2.data import DatasetCatalog

# print(DatasetCatalog.list())


if "name_train" in DatasetCatalog.list():
    DatasetCatalog.remove("name_train")
    MetadataCatalog.remove("name_train")

register_coco_instances("name_train", {},"/path/to/annotations/annotation_train.json", "/path/to/train/images/")


if "name_validation" in DatasetCatalog.list():
    DatasetCatalog.remove("name_validation")
    MetadataCatalog.remove("name_validation")

register_coco_instances("name_validation", {}, "/path/to/annotations/annotation_validation.json", "/path/to/validation/images/")


if "name_test" in DatasetCatalog.list():
    DatasetCatalog.remove("name_test")
    MetadataCatalog.remove("name_test")

register_coco_instances("name_test", {}, "/path/to/annotations/annotation_test.json", "/path/to/test/images/")




print(DatasetCatalog.list())

# Check if the dataset is registered
if "name_train" in DatasetCatalog.list():
    print("Dataset is registered!")
else:
    print("Dataset is NOT registered!")
    
    
    
if "name_validation" in DatasetCatalog.list():
    print("Dataset is registered!")
else:
    print("Dataset is NOT registered!")
    
    
if "name_test" in DatasetCatalog.list():
    print("Dataset is registered!")
else:
    print("Dataset is NOT registered!")

    

