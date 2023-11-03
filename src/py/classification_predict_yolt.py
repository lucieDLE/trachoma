import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn
from torch.utils.data import DataLoader
import monai

from nets import classification
from loaders.tt_dataset import TTDatasetSeg, TrainTransformsFullSeg, EvalTransformsFullSeg
import pickle

from tqdm import tqdm

def replace_last(str, old, new):
    if old not in str:
        return str
    idx = str.rfind(old)
    return str[:idx] + new + str[idx+len(old):]

def main(args):

    if(os.path.splitext(args.csv_test)[1] == ".csv"):        
        df_test = pd.read_csv(args.csv_test)
    else:        
        df_test = pd.read_parquet(args.csv_test)

    NN = getattr(classification, args.nn)

    model = NN.load_from_checkpoint(args.model)
    model.cuda()
    model.eval()
    
    eval_transform = EvalTransformsFullSeg()

    test_ds = monai.data.Dataset(TTDatasetSeg(df_test, mount_point=args.mount_point, img_column=args.img_column, seg_column=args.seg_column, class_column=args.class_column), transform=eval_transform)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)

    pred = []
    probs = []
    softmax = nn.Softmax()

    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        for k in batch:
            batch[k] = batch[k].cuda(non_blocking=True)
        x, X_patches = model(batch)

        x = x.detach().squeeze()

        pred.append(torch.argmax(x).cpu().numpy())
        probs.append(softmax(x).cpu().numpy())


    df_test["pred"] = pred

    out_name = os.path.join(os.path.basename(os.path.dirname(args.model)), os.path.splitext(os.path.basename(args.csv_test))[0] + "_" + os.path.splitext(os.path.basename(args.model))[0] + "_prediction")
    out_name = os.path.join(args.out, out_name)
    out_dir = os.path.dirname(out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing:", out_name)
    df_test.to_csv(out_name + ".csv", index=False)
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

    hparams_group = parser.add_argument_group('Hyperparameters')

    hparams_group.add_argument('--nn', help='Type of PL neural network', type=str, default="MobileYOLT")

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default='./out')

    args = parser.parse_args()
    
    main(args)
