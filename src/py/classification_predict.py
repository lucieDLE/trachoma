import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from nets import classification
from loaders.tt_dataset import TTDataset

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

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

    df_test = pd.read_csv(args.csv_test)
    df_test = remove_labels(df_test, args)

    test_ds = TTDataset(df_test, args.mount_point, img_column=args.img_column, class_column=args.class_column)
    test_data = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    print(df_test['label'].value_counts())

    # if args.nn == "efficientnet_v2s":
    #     model = EfficientnetV2s(args, out_features=args.out_features).load_from_checkpoint(args.model)

    NN = getattr(classification, args.nn)
    model = NN.load_from_checkpoint(args.model)

    model.eval()
    model.cuda()

    with torch.no_grad():
        probs = []
        for idx, (X, Y) in enumerate(tqdm(test_data, total=len(test_data))):
            X = X.cuda()
            pred = model(X)
            probs.append(pred)       


    probs = torch.cat(probs, dim=0)
    predictions = torch.argmax(probs, dim=1)

    probs = probs.cpu().numpy()
    df_test["pred"] = predictions.cpu().numpy()

    print(classification_report(df_test[args.class_column], df_test["pred"]))

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    out_name = os.path.join(args.out, os.path.basename(args.csv_test))
    df_test.to_csv(out_name.replace(".csv", "_prediction.csv"), index=False)

    pickle.dump(probs, open(out_name.replace(".csv", "_probs.pickle"), 'wb'))

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT classification prediction')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv_test', help='Test csv', type=str, required=True)
    input_group.add_argument('--model', help='Model of trained model', type=str, required=True)
    input_group.add_argument('--img_column', help='Name of the image column on the csv', type=str, default="image")
    input_group.add_argument('--class_column', help='Name of the class column on the csv', type=str, default="patch_class")
    input_group.add_argument('--label_column', help='tag column name in csv, containing actual name', type=str, default="label")
    input_group.add_argument('--drop_labels', type=str, default=None, nargs='+', help='drop labels in dataframe')    
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s")

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")
    
    args = parser.parse_args()

    main(args)
