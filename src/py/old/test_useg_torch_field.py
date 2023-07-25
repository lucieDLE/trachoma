import os
import sys

import pandas as pd
import SimpleITK as sitk 
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    ScaleIntensityd,
    ToTensord
)

import argparse
from collections import namedtuple

import tensorflow as tf

import resample
import poly_fit as pf
import pickle
from create_stack import *

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DatasetGenerator(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.loc[idx]
        img = row["img"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img)).reshape((512, 512, 3)).astype(float)

        return {"img": img_np, "idx": idx}
    

def main():    
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    csv_fn = "/work/jprieto/data/remote/EGower/jprieto/eyes_cropped_resampled_512_seg_test.csv"
    test_df = pd.read_csv(csv_fn)
    
    test_transforms = Compose(
        [
            AsChannelFirstd(keys=["img"]),            
            ScaleIntensityd(keys=["img"]),
            ToTensord(keys=["img"])
        ]
    )

    # create a training data loader
    test_ds = monai.data.Dataset(data=DatasetGenerator(test_df), transform=test_transforms)    

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=8,
        collate_fn=list_data_collate,
        pin_memory=True,
    )

    model = monai.networks.nets.UNet(spatial_dims=2, in_channels=3, out_channels=4, channels=(16, 32, 64, 128, 256, 512, 1024), strides=(2, 2, 2, 2, 2, 2), num_res_units=4)

    model_path = "train/torch_unet_train_01252022/model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    predictions = []
    with torch.no_grad():
        model.eval()
        
        for idx, test_data in enumerate(test_loader):
            
            img_fn = test_df.loc[idx]["img"]
            out_fn = os.path.join("segmentation_predict_torch_model01252022", os.path.basename(img_fn))
            
            test_images = test_data["img"].to(device)
            test_outputs = model(test_images)
            test_outputs = torch.argmax(test_outputs, dim=1, keepdim=True)
            test_outputs = torch.permute(test_outputs[0], (1, 2, 0))
            test_outputs = test_outputs.cpu().numpy()

            print("Writing", out_fn)
            predictions.append(out_fn)

            sitk.WriteImage(sitk.GetImageFromArray(test_outputs, isVector=True), out_fn)

    test_df["pred"] = predictions
    test_df.to_csv(csv_fn.replace(".csv", "_torch_prediction.csv"), index=False)


if __name__ == "__main__":
    
    main()





def main(args): 

    model_seg = tf.keras.models.load_model(args.seg_model)
    model_predict = tf.keras.models.load_model(args.predict_model)

    img_out = []

    if args.csv:

        with open(args.csv) as csvfile:
            df = pd.read_csv(csvfile)

            for idx, row in df.iterrows():
                img_out.append({'img': row["img"]})

    else:
        img_out.append({'img': args.img})

    pred = []
    x_a = []
    x_v = []
    x_s = []
    x_v_p = []

    for obj in img_out:


        img = sitk.ReadImage(obj["img"])  

        out_stack, seg = create_stack(img, model_seg, args)

        out_np_stack = sitk.GetArrayFromImage(out_stack)

        pred_np, x_a_np, x_v_np, x_s_np, x_v_p_np = model_predict.predict(np.array([out_np_stack]).astype(np.float32))


        pred.append(pred_np)
        x_a.append(x_a_np)
        x_v.append(x_v_np)
        x_s.append(x_s_np)
        x_v_p.append(x_v_p_np)

    pred = np.concatenate(pred)
    x_a = np.concatenate(x_a)
    x_v = np.concatenate(x_v)
    x_s = np.concatenate(x_s)
    x_v_p = np.concatenate(x_v_p)
    with open(args.out, 'wb') as f:
        pickle.dump((pred, x_a, x_v, x_s, x_v_p), f)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the image stack from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')

    in_group.add_argument('--csv', type=str, help='CSV file with images. ')
    

    parser.add_argument('--seg_model', type=str, help='Segmentation tensorflow model', default='/work/jprieto/data/remote/EGower/jprieto/trained/eyes_cropped_resampled_512_seg_train_random_rot_09072021')
    parser.add_argument('--predict_model', type=str, help='Prediction model tensorflow model', default='/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_resnet_att_17012022')
    #previous predict stacks model /work/jprieto/data/remote/EGower/jprieto/trained/stack_training_resnet_att_06012022

    parser.add_argument('--stack_size', type=int, help='Size w/h of the image stacks/frames', default=448)  
    parser.add_argument('--stack_samples', type=int, help='Stack samples', default=16)
    parser.add_argument('--out', type=str, help='Output prediction', default="out.pickle")

    args = parser.parse_args()
    main(args)
