import logging
import os
import sys
import tempfile
from glob import glob
import math

from sklearn.model_selection import train_test_split
import pandas as pd
import SimpleITK as sitk 
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    RandRotated,
    ScaleIntensityd,
    ToTensord,
    EnsureType,
    Activations, 
    AsDiscrete
)
from monai.visualize import plot_2d_or_3d_image

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
