import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.segmentation import TTUNet
from loaders.tt_dataset import TTDatasetSeg, InTransformsSeg, OutTransformsSeg
from callbacks.logger import SegImageLogger

import SimpleITK as sitk

from sklearn.utils import class_weight

def main(args):
    
    test_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_seg_test_202208.csv')
    img_column = "img_path"
    df_test = pd.read_csv(test_fn)

    test_ds = TTDataset(df_test, args.mount_point, img_column=img_column, transform=InTransformsSeg())

    resize_transform = Resize(spatial_size=[512, 512], mode='nearest')
    out_transform = OutTransformsSeg()
    
    model = TTUNet(args, out_channels=4).load_from_checkpoint(args.model)

    for idx, img in tqdm(enumerate(test_ds), total=len(test_ds)):
        img = resize_transform(img)
        img = model(img)
        img = resize_transform.inverse(img)
        img = out_transform(img)

        img = sitk.GetImageFromArray(img)

        img_fn = os.path.join(args.out, test_ds.loc[idx]['img_column'])

        out_dir = os.path.dirname(img_fn)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT segmentation Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=1000)    
    parser.add_argument('--patience', help='Patience number for early stopping', type=int, default=100)
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    # parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s")    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="segmentation_unet")


    args = parser.parse_args()

    main(args)
