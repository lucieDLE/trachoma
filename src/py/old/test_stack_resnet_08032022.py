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
        self.feat = models.resnet50(pretrained=True)
        self.feat.fc = Identity()
        self.conv = nn.Conv2d(2048, 512, (2, 2), stride=2)
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x):
        x = self.feat.conv1(x)
        x = self.feat.bn1(x)
        x = self.feat.relu(x)
        x = self.feat.maxpool(x)
        x = self.feat.layer1(x)
        x = self.feat.layer2(x)
        x = self.feat.layer3(x)
        x = self.feat.layer4(x)
        x = self.conv(x)
        x = self.avg(x)
        x = torch.flatten(x, start_dim=1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # Args:
        #     x: Tensor, shape [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TTNet(nn.Module):
    def __init__(self):
        super(TTNet, self).__init__()

        
        self.Features = Features()
        self.TimeDistributed = TimeDistributed(self.Features)
        self.WV = nn.Linear(512, 256)
        self.PE = PositionalEncoding(d_model=256, max_len=16)
        self.Attention = SelfAttention(512, 128)
        self.Prediction = nn.Linear(256, 2)
        
 
    def forward(self, x):

        x = self.TimeDistributed(x)

        x_v = self.WV(x)
        x_v = self.PE(x_v)

        x_a, w_a, w_s = self.Attention(x, x_v)

        x = self.Prediction(x_a)
        x = nn.LogSoftmax(dim=1)(x)

        return x


class DatasetGenerator(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.loc[idx]            
        img = os.path.join("/work/jprieto/data/remote/EGower/", row["img"])
        sev = row["class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))


        _t, xs, ys, _c = img_np.shape
        xo = (xs - 448)//2
        yo = (ys - 448)//2

        img_np = img_np[:, xo:xo + 448, yo:yo + 448,:]
        img_np = np.transpose(img_np, (0, 3, 1, 2))

        return {"img": img_np, "class": sev}


model_path = "train/torch_stack_resnet_torch_08032022/model.pt"
device = torch.device("cuda")

model = TTNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/hinashah/Analysis_Set_01132022/trachoma_normals_healthy_sev123_epi_stack_16_768_test.csv"

test_df = pd.read_csv(csv_path_stacks).replace("/work/jprieto/data/remote/EGower/", "", regex=True)
test_df['class'] = (test_df['class'] >= 1).astype(int)

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
            test_outputs = model(test_images)
            test_outputs = torch.argmax(test_outputs, dim=1)
            test_outputs = test_outputs.cpu().numpy()            

            predictions.append(test_outputs[0])


output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_epi_stack_16_768_test_08032022.pickle'), 'wb') as f:
    pickle.dump(predictions, f)

test_df["pred"] = predictions
test_df.to_csv("/work/jprieto/data/remote/EGower/jprieto/test_output/trachoma_normals_healthy_sev123_epi_stack_16_768_test_08032022.csv", index=False)

print(classification_report(test_df["class"], test_df["pred"]))