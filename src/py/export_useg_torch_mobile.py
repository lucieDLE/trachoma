import logging
import os
import sys
import math

import pandas as pd
import numpy as np

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from nets.segmentation import TTUNet, TTUSegTorch
from loaders.tt_dataset import TTDatasetSeg, TrainTransformsSeg, ExportTransformsSeg

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():    

    # csv_fn = "Analysis_Set_202208/trachoma_bsl_mtss_besrat_field_seg_train_202208_eval.csv"
    # test_df = pd.read_csv(csv_fn)

    # # create a training data loader
    # test_ds = monai.data.Dataset(data=TTDatasetSeg(test_df, mount_point="./", img_column="img_path", seg_column="seg_path"), transform=ExportTransformsSeg())

    # example = test_ds[0]["img"]/255.0
    # print(example.shape)

    model_path = "train/Analysis_Set_202208/segmentation_unet/v3/epoch=490-val_loss=0.07.ckpt"  
    model = TTUSegTorch(TTUNet(out_channels=4).load_from_checkpoint(model_path).model)
    model.eval()

    example = torch.rand(1, 3, 512, 512)
    out = model(example)
    print(out.shape)
    
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(model_path.replace(os.path.splitext(model_path)[1], ".ptl"))
            


if __name__ == "__main__":
    
    main()
