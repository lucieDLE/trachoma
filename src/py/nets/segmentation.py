import math
import numpy as np 

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
import torchmetrics

import monai
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution
from monai.metrics import DiceMetric

from monai.transforms import (
    ToTensord
)

import pytorch_lightning as pl

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

class TTUSeg(nn.Module):
    def __init__(self, unet):
        super(TTUSeg, self).__init__()
        self.unet = unet

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.unet(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.argmax(x, dim=-1, keepdim=True)
        x = x.type(torch.uint8)
        x = torch.squeeze(x, 0)
        return x

class TTUSegTorch(nn.Module):
    def __init__(self, unet):
        super(TTUSegTorch, self).__init__()
        self.unet = unet

    def forward(self, x):        
        x = self.unet(x)        
        x = torch.argmax(x, dim=1, keepdim=True)
        x = x.type(torch.uint8)
        return x

class TTUNet(pl.LightningModule):
    def __init__(self, out_channels=4, **kwargs):
        super(TTUNet, self).__init__()        
        
        self.save_hyperparameters()
            
        # self.loss = monai.losses.DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
        self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, 
            ce_weight=torch.tensor([0.1, 1, 1, 4]), lambda_dice=1.0, lambda_ce=1.0)

        self.accuracy = torchmetrics.Accuracy()

        self.model = monai.networks.nets.UNet(spatial_dims=2, in_channels=3, out_channels=self.hparams.out_channels, channels=(16, 32, 64, 128, 256, 512, 1024), strides=(2, 2, 2, 2, 2, 2), num_res_units=4)

        # self.metric = DiceMetric(include_background=True, reduction="mean")   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        
        x = self.model(x)
        x = torch.argmax(x, dim=1, keepdim=True)
        return x

    def training_step(self, train_batch, batch_idx):
        
        x = train_batch["img"]
        y = train_batch["seg"]
        
        y = y.to(torch.int64)
        x = self.model(x)

        loss = self.loss(x, y)
        
        batch_size = x.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)        

        # x = torch.argmax(x, dim=1, keepdim=True)
        # self.accuracy(x.reshape(-1, 1), y.reshape(-1, 1))
        # self.log("train_acc", self.accuracy, batch_size=batch_size)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["img"]
        y = val_batch["seg"]
        
        y = y.to(torch.int64)
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        batch_size = x.shape[0]
        self.log('val_loss', loss, batch_size=batch_size)
        
        # x = torch.argmax(x, dim=1, keepdim=True)
        # self.accuracy(x.reshape(-1, 1), y.reshape(-1, 1))
        # self.log("val_acc", self.accuracy, batch_size=batch_size)

class RandomRotate(nn.Module):
    def __init__(self, degrees, keys, interpolation, prob=0.5):        
        self.degrees = degrees
        self.keys = keys
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, X):

        if self.prob > torch.rand(1):

            angle = torch.rand(1)*self.degrees
            
            for k, interp in zip(self.keys, self.interpolation):
                X[k] = transforms.functional.rotate(X[k], angle=angle.item(), interpolation=interp)

            return X

        return X

class RandomFlip(nn.Module):
    def __init__(self, keys, prob=0.5):
        self.keys = keys   
        self.prob = prob

    def __call__(self, X):

        if self.prob > torch.rand(1):
            for k in self.keys:
                X[k] = transforms.functional.hflip(X[k])
            return X

        return X

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        return self.model(x)

class MobileYOLO(pl.LightningModule):
    def __init__(self, **kwargs):
        super(MobileYOLO, self).__init__()        
        
        self.save_hyperparameters()

        self.model = nn.Sequential(
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            ProjectionHead(input_dim=1280, hidden_dim=1280, output_dim=4)#h,w,i,j
            )

        self.resize_bb = transforms.Resize(self.hparams.size_bb)

        self.train_transform = transforms.Compose(
            [
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)
            ]
        )

        self.spatial_train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )

        self.loss_fn = torch.nn.SmoothL1Loss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        if len(ij) > 0:

            bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
            bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
            bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
            bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def forward(self, X):        
        return self.model(self.resize_bb(X).detach().clone())

    def training_step(self, train_batch, batch_idx):

        train_batch = self.spatial_train_transform(train_batch)
        
        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in train_batch["seg"]]).to(self.device)
        
        x_bb = self(self.train_transform(train_batch["img"]))

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in val_batch["seg"]]).to(self.device)

        x_bb = self(val_batch["img"])

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('val_loss', loss, sync_dist=True)
        

    def test_step(self, test_batch, batch_idx):
        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in test_batch["seg"]]).to(self.device)

        x_bb = self(test_batch["img"])

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('test_loss', loss, sync_dist=True)