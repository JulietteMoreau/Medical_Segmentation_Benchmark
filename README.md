# Medical Segmentation Architectures

Five deep learning segmentation architectures, initially designed for stroke lesion segmentation. It includes U-Net 2D, cGAN, Mask R-CNN, UNETR and U-Net 3D. All scripts, expect for Mask R-CNN which implementation is particular, were used with the versions of the libraries present in `requirements.txt`. They can be installed using

```
pip install -r requirements.txt
```



# Data organization and preprocessings

The data shall be organized as the following tree to use the different codes. Folders 2D and 3D may not be in the same folder.


```
data/
├── 2D/
│   ├── image/
│   │   ├── train/
│   │   │   └── img1.jpg
│   │   ├── validation/
│   │   │   └── img2.jpg
│   │   └── test/
│   │       └── img3.jpg
│   └── reference/
│       ├── train/
│       │   └── img1.png
│       ├── validation/
│       │   └── img2.png
│       └── test/
│           └── img3.png
├── 3D/
│   ├── image/
│   │   ├── train/
│   │   │   └── img1.nii.gz
│   │   ├── validation/
│   │   │   └── img2.nii.gz
│   │   └── test/
│   │       └── img3.nii.gz
│   └── reference/
│       ├── train/
│       │   └── img1.nii.gz
│       ├── validation/
│       │   └── img2.nii.gz
│       └── test/
│           └── img3.nii.gz
```



The 2D slices are produced from the 3D images thanks to [med2image](https://github.com/FNNDSC/med2image) and the following code lines.

```
# medical images
med2image -i directory/image.nii.gz -d /output/dir
# reference masks
med2image -i directory/image.nii.gz -d /output/dir -t png
```

In the original usage of the codes, only the slices with a reference where kept, but the models can also be trained with healthy images.
All trainings are implemented for three sets: train, validation and test. An early-stopping process is implmented on the validation set to stop trining if the validation loss does not improve for some epochs (set as patience parameter).

# U-Net 2D

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9,
2015, proceedings, part III 18, pages 234–241. Springer, 2015

The U-Net architecture is the state of the art architecture in medical image segmentation. This version is a classical version with four downsamplings and upsamplings with skip connections in between. You shall adapt the path in the `run.sh` file to run a training. Once you have your final model, you can evaluate the performance with the `evaluation_checkpoint.py` script, in changing the paths. The `evaluation_checkpoint_3d.py` allows to get the performances based on the 3D volumes.


# cGAN (conditional Generative Adversarial Network)

Biting Yu, Luping Zhou, Lei Wang, Jurgen Fripp, and Pierrick Bourgeat. 3D cGAN based cross-modality MR image synthesis for brain tumor segmentation. In 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018), pages 626–630. IEEE, 2018.

The cGAN is based on the previous U-Net with an additional discriminating branch which role is to distinguish real masks from fake masks to improve the predicted masks. You shall adapt the path in the `run.sh` file to run a training. Once you have your final model, you can evaluate the performance with the `evaluation_checkpoint.py` script, in changing the paths.

# Mask R-CNN

Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.

The Mask R-CNN is an architecture that proposes a segmentation preceded by an object detection to try to improve the segmentation. The proposed version is based on [detectron2](https://github.com/facebookresearch/detectron2) implementation with a modification for the early-stopping process. You shall download and install [detectron2](https://github.com/facebookresearch/detectron2), modify the script LossEvalHook.py to implement the early-stopping, save your datasets in the COCO format which is used for this implementation with `prepare_data/convert_to_COCO_1_json.py`, and adapt the config file with the corresponding paths before launching the training with `run.sh`. The evaluation is also different and the performance measurements can be done with `evaluation/calcul_metriques_coupes.py`, while the segmentation visualization is made with `evaluation/lecture_segmentation.py`, both in changing the paths, after infering on the set of interest using the second line in `run.sh` and a adapted config file.

# UNETR

Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger R Roth, and Daguang Xu. UNETR: Transformers for 3D medical image segmentation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 574–584, 2022.

The UNETR mixes the U-Net and a Transformer. You shall adapt the path in the `run.sh` file to run a training. Once you have your final model, you can evaluate the performance with the `evaluation_checkpoint.py` script, in changing the paths.

# U-Net 3D

Özgün Çiçek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3D U-Net: learning dense volumetric segmentation from sparse annotation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International
Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19, pages 424–432. Springer, 2016.

The U-Net 3D is adapted from the U-Net to segment the 3D images. You shall adapt the path in the `run.sh` file to run a training. Once you have your final model, you can evaluate the performance with the `evaluation_checkpoint.py` script, in changing the paths.

