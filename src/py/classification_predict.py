import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from nets.classification import EfficientnetV2s
from loaders.tt_dataset import TTDataset

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

def main(args):
    
    test_fn = os.path.join(args.mount_point, args.csv)

    df_test = pd.read_csv(test_fn)
    df_test.drop(df_test[df_test[args.class_column].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)

    unique_classes = np.sort(np.unique(df_test[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_test[args.class_column]))

    # class_replace = {}
    # for cn, cl in enumerate(unique_classes):
    #     class_replace[cl] = cn
    # print(unique_classes, unique_class_weights, class_replace)

    # df_test[args.class_column] = df_test[args.class_column].replace(class_replace)
    
    df_test[args.class_column] = df_test[args.class_column].replace({'Healthy': 0, 'Epilation': 1, 'TT': 2})

    test_ds = TTDataset(df_test, args.mount_point, img_column=args.img_column, class_column=args.class_column)
    test_data = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    if args.nn == "efficientnet_v2s":
        model = EfficientnetV2s(args, out_features=args.out_features).load_from_checkpoint(args.model)

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

    df_test.to_csv(args.csv.replace(".csv", "_prediction.csv"), index=False)

    pickle.dump(probs, open(args.csv.replace(".csv", "_probs.pickle"), 'wb'))

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT classification prediction')    
    parser.add_argument('--csv', help='Test csv', type=str, required=True)
    parser.add_argument('--img_column', help='Name of the image column on the csv', type=str, default="image")
    parser.add_argument('--class_column', help='Name of the class column on the csv', type=str, default="patch_class")
    parser.add_argument('--model', help='Model of trained model', type=str, required=True)    
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--out_features', help='Out features', type=int, default=4)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s")        


    args = parser.parse_args()

    main(args)
