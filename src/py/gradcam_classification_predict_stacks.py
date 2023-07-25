import argparse

import os
import pandas as pd
import numpy as np 

import SimpleITK as sitk

import torch
from torch.utils.data import DataLoader

from nets.classification import EfficientnetV2s, EfficientnetV2sStacks, TimeDistributed
from loaders.tt_dataset import TTDatasetStacks

from torchvision import transforms

from tqdm import tqdm
import pickle

from sklearn.metrics import classification_report

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

def main(args):
    
    test_fn = os.path.join(args.mount_point, args.csv)

    class_column = args.class_column
    img_column = args.img_column
    
    df_test = pd.read_csv(test_fn)    
    
    test_ds = TTDatasetStacks(df_test, mount_point=args.mount_point, img_column=img_column, class_column=class_column)
    test_data = DataLoader(test_ds, shuffle=False, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)    

    if args.nn == "efficientnet_v2s_stacks":
        model = EfficientnetV2sStacks(args, out_features=args.out_features, features=True).load_from_checkpoint(args.model)
        model.features = True
    
    model.eval()
    model.cuda()

    test_transform = TimeDistributed(torch.nn.Sequential(            
        transforms.CenterCrop(448)
    ))
    test_transform.cuda()

    model_patches = model.model_patches
    target_layers = [model_patches[1]]

    cam = GradCAM(model=model_patches, target_layers=target_layers, use_cuda=True)

    
    for idx, (X, Y) in enumerate(tqdm(test_data, total=len(test_data))):
        
        out_fn = os.path.join(args.out, df_test.loc[idx][img_column])

        X = X.cuda(non_blocking=True).contiguous()
        X = test_transform(X)

        tt_cam = []
        for x_frame in X[0]:
            
            grayscale_cam = cam(input_tensor=x_frame.unsqueeze(dim=0), targets=None, eigen_smooth=True)
            tt_cam.append(np.expand_dims(grayscale_cam.transpose((1, 2, 0)), axis=0))
        tt_cam = np.concatenate(tt_cam, axis=0)

        out_dir = os.path.dirname(out_fn)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir
                )
        sitk.WriteImage(sitk.GetImageFromArray(tt_cam, isVector=True), out_fn)

    

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT classification Predict stacks')    
    parser.add_argument('--csv', help='CSV file for prediction', type=str, required=True)        
    parser.add_argument('--class_column', help='Class column name', type=str, default="class")
    parser.add_argument('--img_column', help='Image column name', type=str, default="image")
    parser.add_argument('--model', help='Model path to continue training', type=str, required=True)        
    parser.add_argument('--out_features', help='Output number of classes', type=int, default=2)        
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)    
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s_stacks")    
    parser.add_argument('--out', help='Output directory', type=str, default="./cam")
    


    args = parser.parse_args()

    main(args)
