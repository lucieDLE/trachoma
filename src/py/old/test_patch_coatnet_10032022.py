import os
import sys
from glob import glob
import math

import pandas as pd
import SimpleITK as sitk 
import numpy as np
from tqdm import tqdm

import torch
import pickle

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models

from sklearn.metrics import classification_report

from coatnet import CoAtNet

from monai.data import list_data_collate
import monai
from monai.transforms import (    
    Compose,
    ScaleIntensityd,
    ToTensord
)

from torchvision import transforms as T

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
        

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):
        
        score = self.V(nn.Tanh()(self.W1(query)))
        
        # score = nn.Sigmoid()(score)
        # sum_score = torch.sum(score, 1, keepdim=True)
        # attention_weights = score / sum_score
        attention_weights = nn.Softmax(dim=1)(score)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights, score


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()

        num_blocks = [2, 2, 3, 5, 2]            # L
        channels = [64, 96, 192, 384, 768]      # D
        block_types=['C', 'T', 'T', 'T']        # 'C' for MBConv, 'T' for Transformer

        self.feat = CoAtNet((448, 448), 3, num_blocks, channels, block_types=block_types, num_classes=3)

    def forward(self, x):
        x = self.feat(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x

class TTNet(nn.Module):
    def __init__(self):
        super(TTNet, self).__init__()
        
        self.Features = Features()        
        
 
    def forward(self, x):
        return self.Features(x)


class DatasetGenerator(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.loc[idx]            
        img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
        sev = row["patch_class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))
        img_np = np.transpose(img_np, (2, 0, 1))#channels first

        return {"img": img_np, "patch_class": sev}


class TestTransforms(nn.Module):
    def __init__(self):
        super(TestTransforms, self).__init__()

        self.T = T.Compose([            
            T.CenterCrop(448)
        ])
 
    def forward(self, x):

        return self.T(x)

model_path = "train/train_patch_coatnet_10032022/model.pt"
device = torch.device("cuda")

model = TTNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

model_test_transforms = TestTransforms()

csv_path = "/work/jprieto/data/remote/EGower/jprieto/hinashah/Analysis_Set_01132022/trachoma_normals_healthy_sev123_epi_patches_test.csv"

test_df = pd.read_csv(csv_path).replace("/work/jprieto/data/remote/EGower/", "", regex=True)
test_df.drop(test_df[test_df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
test_df = test_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
test_df = test_df.reset_index(drop=True)

test_transforms = Compose(
    [
        ScaleIntensityd(keys=["img"]),
        ToTensord(keys=["img"])
    ]
)

test_ds = monai.data.Dataset(data=DatasetGenerator(test_df), transform=test_transforms)

test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=8,
        collate_fn=list_data_collate
    )

model.eval()

with torch.no_grad():
    pbar = tqdm(test_loader)
    predictions = []
    model.eval()
    for test_data in pbar:
            
            test_images = test_data["img"].to(device)
            test_images = model_test_transforms(test_images)
            test_outputs = model(test_images)
            test_outputs = torch.argmax(test_outputs, dim=1)
            test_outputs = test_outputs.cpu().numpy()            

            predictions.append(test_outputs[0])


output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_epi_stack_16_768_test_09032022.pickle'), 'wb') as f:
    pickle.dump(predictions, f)

test_df["pred"] = predictions
test_df.to_csv("/work/jprieto/data/remote/EGower/jprieto/test_output/trachoma_normals_healthy_sev123_epi_stack_16_768_test_08032022.csv", index=False)

print(classification_report(test_df["patch_class"], test_df["pred"]))