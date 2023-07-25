import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from loaders.tt_dataset import TTDataModuleSeg, TrainTransformsSeg, EvalTransformsSeg

def main(args):

    train_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_seg_train_202208_train.csv')
    valid_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_seg_train_202208_eval.csv')
    test_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_seg_test_202208.csv')

    df_train = pd.read_csv(train_fn)
    df_val = pd.read_csv(valid_fn)    
    df_test = pd.read_csv(test_fn)
    
    ttdata = TTDataModuleSeg(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', seg_column="seg_path", train_transform=TrainTransformsSeg(), valid_transform=EvalTransformsSeg(), test_transform=EvalTransformsSeg())
    ttdata.setup()
    for batch in ttdata.val_dataloader():
        print(batch)
        # img, seg = batch
        # print(img.shape, img.dtype)
        # print(seg.shape, seg.dtype)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TT test seg loader')
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=1)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)    

    args = parser.parse_args()

    main(args)