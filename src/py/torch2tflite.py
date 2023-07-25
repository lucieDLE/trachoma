import os
import sys

import onnx
from onnx_tf.backend import prepare

import torch
import torch.onnx
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn

import tensorflow as tf 
import pandas as pd
import numpy as np
import nrrd


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
from nets.segmentation import TTUNet, TTUSeg

device = 'cpu'
# model_path = "train/torch_unet_train_01252022/model.pt"
model_path = "/work/jprieto/data/remote/EGower/jprieto/train/Analysis_Set_202208/segmentation_unet/v3/epoch=490-val_loss=0.07.ckpt"
# model.unet.load_state_dict(torch.load(model_path, map_location=device))    
model = TTUSeg(TTUNet(out_channels=4).load_from_checkpoint(model_path).model)
model.eval()

#Input to the model
batch_size = 1
x = torch.randn(512, 512, 3)

torch_out = model(x)
print(torch_out.shape)

onnx_model_path = model_path.replace(".ckpt", ".onnx")

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_model_path,   # where to save the model (can be a file or file-like object)                  
                  do_constant_folding=True,
                  export_params=True, 
                  opset_version=16,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_model_path = onnx_model_path.replace('.onnx', '_saved_model')
tf_rep.export_graph(tf_model_path)


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

tflite_model_path = tf_model_path + '.tflite'
# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
