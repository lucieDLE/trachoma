import os
import sys

import onnx
from onnx_tf.backend import prepare
# from onnx2keras import onnx_to_keras

import torch
import torch.onnx
import torch.optim as optim
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import tensorflow as tf 

import argparse

import os
import numpy as np 

from torchvision import models
from torchvision import ops
import pickle
import monai

def main(args):

    with torch.no_grad():

        # model = models.efficientnet_v2_s().cuda()
        # model.eval()

        # model = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=3, pretrained=True).cuda()
        # model = monai.networks.nets.EfficientNetBN('efficientnet-b0').cuda()
        feat = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=3, pretrained=True)

        # self.feat = TimeDistributed(feat)
        # self.pool = AveragePool1D(dim=1)
        # self.pred = nn.Linear(2048, self.hparams.out_features)

        model = nn.Sequential(
            feat.layer0,
            feat.layer1,
            feat.layer2,
            feat.layer3,
            feat.layer4,
            ops.Conv2dNormActivation(2048, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1536, out_features=3, bias=True)
        )
        model.cuda()
        model.eval()
        x = torch.randn(1, 3, 448, 448, dtype=torch.float32).cuda()
        # torch_out = model(x)

        # model_features_script_module = torch.jit.trace(model_features, x)
        # model_traced = torch.jit.trace(model, x)
        onnx_model_path = "test_model.onnx"
        # Export the model
        torch.onnx.export(model,               # model being run
                          x,                         # model input (or a tuple for multiple inputs)
                          onnx_model_path,   # where to save the model (can be a file or file-like object)
                          opset_version=args.opset_version,
                          do_constant_folding=True,
                          export_params=True,
                          input_names = ['input_1'],   # the model's input names
                          output_names = ['output_1']) # the model's output names)


        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model, device='CUDA', logging_level='DEBUG')
        tf_model_path = onnx_model_path.replace('.onnx', '_saved_model')
        tf_rep.export_graph(tf_model_path)

        # tf_model = tf.saved_model.load(tf_model_path)
        # out = tf_model(input=tf.random.normal((1, 448, 448, 3)))
        # print(out)
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float16]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [
            tf.lite.OpsSet.SELECT_TF_OPS,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        # target_spec.experimental_supported_backends = ["GPU"]
        target_spec.supported_types = [tf.float16]

        converter.target_spec = target_spec

        tflite_model = converter.convert()

        tflite_model_path = tf_model_path + '.tflite'
        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Export TT Stacks model')
    
    # parser.add_argument('--model', help='Model path to export', type=str, required=True)
    parser.add_argument('--opset_version', help='opset_version -> check doc from torch.onnx.export', type=int, default=16)
    

    args = parser.parse_args()

    main(args)
