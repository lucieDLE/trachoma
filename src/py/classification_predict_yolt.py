import argparse

import math
import os
import pandas as pd
import numpy as np 
from monai.data.utils import pad_list_data_collate
import pdb
import torch
from torch import nn
from torch.utils.data import DataLoader
import monai

from nets import classification
from loaders.tt_dataset import TTDatasetSeg, TrainTransformsFullSeg, EvalTransformsFullSeg, BBXImageTestTransform, TTDatasetPatch
import pickle

from tqdm import tqdm

def replace_last(str, old, new):
    if old not in str:
        return str
    idx = str.rfind(old)
    return str[:idx] + new + str[idx+len(old):]

def remove_labels(df, args):

    if args.drop_labels is not None:
        df = df[ ~ df[args.label_column].isin(args.drop_labels)]

    if args.concat_labels is not None:
        replacement_val = df.loc[ df['label'] == args.concat_labels[0]]['class'].unique()
        df.loc[ df['label'].isin(args.concat_labels), "class" ] = replacement_val[0]

    unique_classes = sorted(df[args.class_column].unique())
    class_mapping = {value: idx for idx, value in enumerate(unique_classes)}

    df[args.class_column] = df[args.class_column].map(class_mapping)
    print(f"Kept Classes : {df[args.label_column].unique()}, {class_mapping}")
    return df


def main(args):

    if(os.path.splitext(args.csv_test)[1] == ".csv"):        
        df_test = pd.read_csv(args.csv_test)
    else:        
        df_test = pd.read_parquet(args.csv_test)

    df_test = remove_labels(df_test, args)
    NN = getattr(classification, args.nn)

    model = NN.load_from_checkpoint(args.model)
    model.cuda()
    model.eval()
    
    eval_transform = EvalTransformsFullSeg()

    test_ds = monai.data.Dataset(TTDatasetSeg(df_test, mount_point=args.mount_point, img_column=args.img_column, seg_column=args.seg_column, class_column=args.class_column), transform=eval_transform)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers,pin_memory=False, drop_last=True, collate_fn=pad_list_data_collate)

    # test_ds = monai.data.Dataset(TTDatasetPatch(df_test, mount_point=args.mount_point, img_column=args.img_column, class_column=args.class_column,patch_size = args.patch_size,transform=BBXImageTestTransform()))
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1,pin_memory=False, drop_last=True)

    pred,  features, features_v = [], [], []
    probs = []
    softmax = nn.Softmax()
    gt = [] 
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True)
            x, _, x_a, x_v = model(batch)
            y = batch['labels']

            x = x.detach().squeeze()
            features.append(x_a)
            features_v.append(x_v)

            gt.append(y.cpu().numpy())
            pred.append(torch.argmax(x).cpu().numpy())
            probs.append(softmax(x).cpu().numpy())

    features = torch.cat(features, dim=0).cpu().numpy()
    features_v = torch.cat(features_v, dim=0).cpu().numpy()
    df_sorted = df_test.sort_values(by=['filename', 'class'], ascending=[True, False])
    df_sorted = df_sorted.drop_duplicates(subset='filename')
    df_sorted["pred"] = pred
    df_sorted['gt'] = gt

    out_name = os.path.join(os.path.basename(os.path.dirname(args.model)), os.path.splitext(os.path.basename(args.csv_test))[0] + "_" + os.path.splitext(os.path.basename(args.model))[0] + "_prediction")
    out_name = os.path.join(args.out, out_name)
    out_dir = os.path.dirname(out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing:", out_name)
    df_sorted.to_csv(out_name + ".csv", index=False)
    pickle.dump(probs, open(out_name + ".pickle", 'wb'))
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Surgery prediction prediction')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model for inference', type=str, required=True)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')
    input_group.add_argument('--seg_column', type=str, default="seg_path", help='Name of segmentation column in csv')
    input_group.add_argument('--class_column', type=str, default="class", help='Name of class column in csv')
    input_group.add_argument('--label_column', help='tag column name in csv, containing actual name', type=str, default="label")
    input_group.add_argument('--drop_labels', type=str, default=None, nargs='+', help='drop labels in dataframe')
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--patch_size', help='Size of the patch', nargs='+', type=int, default=(256, 256))
    hparams_group.add_argument('--num_patches', help='Number of patches to extract', type=int, default=5)
    hparams_group.add_argument('--pad', help='Pad the bounding box', type=float, default=0.1)
    hparams_group.add_argument('--square_pad', help='how to pad the image', type=int, default=0)

    hparams_group.add_argument('--nn', help='Type of PL neural network', type=str, default="MobileYOLT")

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default='./out')

    args = parser.parse_args()
    
    main(args)
