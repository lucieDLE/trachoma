import argparse
import monai

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.segmentation import TTUNet
from loaders.tt_dataset import TTDataset, EvalTransformsSeg, InTransformsSeg, OutTransformsSeg
from callbacks.logger import SegImageLogger
import pdb
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.utils import class_weight
from monai.transforms import Resize
def main(args):
    
    df_test = pd.read_csv(args.csv_test)

    test_ds = TTDataset(df_test, mount_point=args.mount_point, img_column=args.img_column, transform=InTransformsSeg())

    resize_transform = Resize(spatial_size=[512, 512], mode='nearest')
    out_transform = OutTransformsSeg()
    
    model = TTUNet.load_from_checkpoint(args.model)
    l_files = []
    # pdb.set_trace()

    for idx, img in tqdm(enumerate(test_ds), total=len(test_ds)):

        img = resize_transform(img)
        img = model(img)
        img = resize_transform.inverse(img)
        img = out_transform(img)

        img = sitk.GetImageFromArray(img)

        img_fn = os.path.join(args.out, test_ds.loc[idx]['img_column'])

        out_dir = os.path.dirname(img_fn)
        l_files.appen(img_fn)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(img_fn)
        writer.UseCompressionOn()
        writer.Execute(img)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT segmentation Training')
    parser.add_argument('--csv_test', type=str, required=True)
    parser.add_argument('--img_column', help='Name of the image column on the csv', type=str, default="image")
    parser.add_argument('--seg_column', type=str, default="seg_path", help='Name of segmentation column in csv')

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
