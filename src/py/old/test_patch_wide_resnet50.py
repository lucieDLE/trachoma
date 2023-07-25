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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length=None, height=None, width=None):
        super(PositionalEncoding, self).__init__()

        if length:
            self.positionalencoding1d(d_model, length)
        elif height and width:
            self.positionalencoding2d(d_model, height, width)

    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """        
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe + x

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

        self.resnet = models.wide_resnet50_2(pretrained=True)
        self.resnet.fc = Identity()
        self.pe = PositionalEncoding(2048, height=2, width=2)
        self.Attention = SelfAttention(2048, 128)
        

    def forward(self, x):
        x = torch.stack([self.resnet(x[:,:,0:224,0:224]), self.resnet(x[:,:,224:,0:224]), self.resnet(x[:,:,0:224,224:]), self.resnet(x[:,:,224:,224:])], dim=2)
        x = x.reshape(-1, 2048, 2, 2)
        x = self.pe(x)
        x = x.reshape(-1, 4, 2048)
        x, w_a, w_s = self.Attention(x, x)
        
        return x

class TTNet(nn.Module):
    def __init__(self):
        super(TTNet, self).__init__()
        
        self.Features = Features()
        self.Prediction = nn.Linear(2048, 3)        
 
    def forward(self, x):

        x_f = self.Features(x)
        x = self.Prediction(x_f)
        x = nn.LogSoftmax(dim=1)(x)
        return x, x_f


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

model_path = "train/train_patch_wide_resnet_15032022/model.pt"
device = torch.device("cuda")

model = TTNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

model_test_transforms = TestTransforms()

csv_path = "/work/jprieto/data/remote/EGower/jprieto/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_patches_test.csv"

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
    features = []


    model.eval()
    for test_data in pbar:
            
            test_images = test_data["img"].to(device)
            test_images = model_test_transforms(test_images)
            test_outputs = model(test_images)

            pred = torch.argmax(test_outputs[0], dim=1).cpu().numpy()
            predictions.append(pred[0])
            features.append(test_outputs[1][0].cpu().numpy())


output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

with open(os.path.join(output_dir, 'Analysis_Set_11012021_trachoma_normals_healthy_sev123_epi_patches_test_wide_resnet_15032022.pickle'), 'wb') as f:
    pickle.dump((predictions, features), f)

test_df["pred"] = predictions
test_df.to_csv("/work/jprieto/data/remote/EGower/jprieto/test_output/Analysis_Set_11012021_trachoma_normals_healthy_sev123_epi_patches_test_wide_resnet_15032022.csv", index=False)

print(classification_report(test_df["patch_class"], test_df["pred"]))