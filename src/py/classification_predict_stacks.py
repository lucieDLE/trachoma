import argparse

import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from nets.classification import EfficientnetV2sStacks, MobileNetV2Stacks, EfficientnetV2sStacksDot
from loaders.tt_dataset import TTDatasetStacks

from tqdm import tqdm
import pickle

from sklearn.metrics import classification_report

def main(args):
    
    test_fn = os.path.join(args.mount_point, args.csv)

    class_column = args.class_column
    img_column = args.img_column
    
    df_test = pd.read_csv(test_fn)    
    
    test_ds = TTDatasetStacks(df_test, mount_point=args.mount_point, img_column=img_column, class_column=class_column)
    test_data = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)    

    if args.nn == "efficientnet_v2s_stacks":
        model = EfficientnetV2sStacks(args, out_features=args.out_features, features=True).load_from_checkpoint(args.model)
        model.features = True
    elif args.nn == "efficientnet_v2s_stacks_dot":
        model = EfficientnetV2sStacksDot(args, out_features=args.out_features, features=True).load_from_checkpoint(args.model)
        model.features = True
    elif args.nn == "mobilenet_v2_stacks":
        model = MobileNetV2Stacks(args, out_features=args.out_features, features=True).load_from_checkpoint(args.model)
        model.features = True
    
    model.eval()
    model.cuda()

    probs = []
    features = []
    scores = []
    features_v = []
    features_v_p = []

    with torch.no_grad():        
        for idx, (X, Y) in enumerate(tqdm(test_data, total=len(test_data))):
            X = X.cuda()
            x, x_a, x_s, x_v, x_v_p = model(X)
            probs.append(x)       
            features.append(x_a)
            scores.append(x_s)
            features_v.append(x_v)
            features_v_p.append(x_v_p)


    probs = torch.cat(probs, dim=0)
    predictions = torch.argmax(probs, dim=1)

    features = torch.cat(features, dim=0).cpu().numpy()
    scores = torch.cat(scores, dim=0).cpu().numpy()
    features_v = torch.cat(features_v, dim=0).cpu().numpy()
    features_v_p = torch.cat(features_v_p, dim=0).cpu().numpy()

    probs = probs.cpu().numpy()
    df_test["pred"] = predictions.cpu().numpy()

    print(classification_report(df_test[args.class_column], df_test["pred"]))

    df_test.to_csv(args.csv.replace(".csv", "_prediction.csv"), index=False)
    
    pickle.dump((probs, features, scores, features_v, features_v_p), open(args.csv.replace(".csv", "_features.pickle"), 'wb'))

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT classification Predict stacks')    
    parser.add_argument('--csv', help='CSV file for prediction', type=str, required=True)        
    parser.add_argument('--class_column', help='Class column name', type=str, default="class")
    parser.add_argument('--img_column', help='Image column name', type=str, default="image")
    parser.add_argument('--model', help='Model path to continue training', type=str, required=True)        
    parser.add_argument('--out_features', help='Output number of classes', type=int, default=2)    
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s_stacks")    
    


    args = parser.parse_args()

    main(args)
