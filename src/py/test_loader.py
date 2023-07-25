import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import EfficientnetV2s
from loaders.tt_dataset import TTDataModule, TrainTransforms, EvalTransforms

from sklearn.utils import class_weight

def main(args):

    train_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_fold0_train.csv')
    valid_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_train_20220422_fold0_test.csv')
    test_fn = os.path.join(args.mount_point, 'Analysis_Set_20220422', 'trachoma_bsl_mtss_besrat_field_patches_test_20220422.csv')

    df_train = pd.read_csv(train_fn)
    df_train.drop(df_train[df_train['patch_class'].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)

    unique_classes = np.sort(np.unique(df_train["patch_class"]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train["patch_class"]))

    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[cl] = cn
    print(unique_classes, unique_class_weights, class_replace)

    df_train["patch_class"] = df_train["patch_class"].replace(class_replace)

    df_val = pd.read_csv(valid_fn)    
    df_val.drop(df_val[df_val['patch_class'].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)
    df_val["patch_class"] = df_val["patch_class"].replace(class_replace)
    
    df_test = pd.read_csv(test_fn)
    df_test["patch_class"] = df_test["patch_class"].replace(class_replace)

    
    ttdata = TTDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='image', class_column="patch_class", train_transform=TrainTransforms(256), valid_transform=EvalTransforms(256), test_transform=EvalTransforms(256))
    ttdata.setup()
    for idx, batch in ttdata.train_dataloader():
        print(batch.shape)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TT test loader')
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=1)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)    

    args = parser.parse_args()

    main(args)
