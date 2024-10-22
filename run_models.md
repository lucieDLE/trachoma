# How to run the models

We want to be able to construct a model that can predict the outcome of the surgery done from an image of the eye. 

## Training on patches

The model takes as input small patches around the eye and predict the outcome of this portion of the eye (Healthy, overcorrected, undercorrected (entropion)...). Each csv file contains the path to patches with the class and name associated. 

Because the patches have very low resolution, we apply upsampling techniques like 
[ESRGAN](https://github.com/xinntao/ESRGAN?tab=readme-ov-file).


```
python classification_train.py  --csv_train patches_train_train.csv --csv_valid patches_train_test.csv --csv_test patches_test.csv --mount_point patch_data_fold/esgran_patch --nn EfficientnetV2s --batch_size 4 --out output_folder --img_column path --patch_class_column class

```

## Training on full-size images

The model takes as input images with segmentations mask of the eye and predict the outcome of the eye (Healthy, PTT, ECA). Each csv file contains the path to the images, the segmentation masks, the class and the name associated. The model will create a sequence of patches around the eye (using the mask) and feed them to the network. 

```
python classification_train_yolt.py --csv_train img_train_train.csv --csv_valid img_train_test.csv  --csv_test img_test.csv  --mount_point img_data_fold --img_column "image path" --seg_column "segmentation path" --class_column class --nn EfficientNetV2SYOLTv2 --out output_folder --patch_size 256 256 --num_patches 6 --pad 0.03  
```


The model can use pre-trained weights from the patches model:
```
python classification_train_yolt.py --csv_train train_train.csv --csv_valid train_test.csv  --csv_test test.csv  --mount_point data_fold --img_column "image path" --seg_column "segmentation path" --class_column class --nn EfficientNetV2SYOLTv2 --out output_folder --patch_size 256 256 --num_patches 6 --pad 0.03  --model_feat_nn EfficientnetV2s  --model_feat efficientNetv2s_checkpoints.ckpt

```