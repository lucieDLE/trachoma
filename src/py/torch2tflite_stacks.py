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
tf.config.optimizer.set_jit(True)

import argparse

import os
import numpy as np 

from nets.classification import EfficientnetV2s, EfficientnetV2sStacks, EfficientnetV2sStacksDot, TTFeatures, TTPrediction, TTStacks
import pickle

def main(args):

    model_path = args.model     

    model = EfficientnetV2sStacksDot.load_from_checkpoint(args.model)
    model.eval()
    # model.cuda()
    # print(model)
    # quit()
    model_features = TTFeatures(model.F.module)
    model_features.eval()
    # model_features.cuda()

    # model_full = TTStacks()
    # model_full.F = model.F
    # model_full.V = model.V
    # model_full.A = model.A
    # model_full.P = model.P
    # model_full.eval()
    # model_full.cuda()

    with torch.no_grad():

        # x = torch.randn(1, 16, 3, 448, 448, dtype=torch.float32).cuda()
        
        # torch_out = model_full(x)

        # model_features_script_module = torch.jit.trace(model_features, x)
        # model_full_traced = torch.jit.trace(model_full, x)
        # model_full_traced_optimized = optimize_for_mobile(model_full_traced)
        # model_full_traced_optimized._save_for_lite_interpreter(model_path.replace(".ckpt", "_full.ptl"))


        #Input to the model
        # model_features_script_module = torch.jit.trace(model_features, x)
        model_features_traced = torch.jit.trace(model.F.module, torch.randn(1, 3, 448, 448, dtype=torch.float32))
        model_features_traced_optimized = optimize_for_mobile(model_features_traced)
        model_features_traced_optimized._save_for_lite_interpreter(model_path.replace(".ckpt", "_features.ptl"))


        onnx_model_path = model_path.replace(".ckpt", "_features.onnx")

        batch_size = 1
        x = torch.randn(1, 448, 448, 3, dtype=torch.float32)
        torch_out = model_features(x)

        model_features_traced = torch.jit.trace(model_features, x)
        # Export the model
        torch.onnx.export(model_features_traced,               # model being run
                          x,                         # model input (or a tuple for multiple inputs)
                          onnx_model_path,   # where to save the model (can be a file or file-like object)
                          opset_version=args.opset_version,
                          do_constant_folding=True,
                          export_params=True,
                          input_names = ['input_1'],   # the model's input names
                          output_names = ['output_1']) # the model's output names)


        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model, logging_level='DEBUG')
        tf_model_path = onnx_model_path.replace('.onnx', '_saved_model')
        tf_rep.export_graph(tf_model_path)

        # tf_model = tf.saved_model.load(tf_model_path)
        # out = tf_model(input=tf.random.normal((1, 448, 448, 3)))
        # print(out)
        # Convert the model

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        target_spec = tf.lite.TargetSpec()
        target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        # target_spec.experimental_supported_backends = ["GPU"]
        target_spec.supported_types = [tf.float16]

        converter.target_spec = target_spec

        print("Converting features!")
        tflite_model = converter.convert()

        tflite_model_path = tf_model_path + '.tflite'
        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        model_prediction = TTPrediction()
        model_prediction.V = model.V
        model_prediction.A = model.A
        model_prediction.P = model.P

        model_prediction.eval()

        x = torch.randn(1, 16, 1536, dtype=torch.float32)
        
        torch_out = model_prediction(x)
        # print(torch_out[0].shape, torch_out[1].shape)


        onnx_model_path = model_path.replace(".ckpt", "_prediction.onnx")

        # Export the model
        model_prediction_traced = torch.jit.trace(model_prediction, x)
        model_prediction_traced_optimized = optimize_for_mobile(model_prediction_traced)
        model_prediction_traced_optimized._save_for_lite_interpreter(model_path.replace(".ckpt", "_prediction.ptl"))
        
        torch.onnx.export(model_prediction_traced,               # model being run
                          x,                         # model input (or a tuple for multiple inputs)
                          onnx_model_path,   # where to save the model (can be a file or file-like object)
                          opset_version=args.opset_version,
                          do_constant_folding=True,
                          export_params=True,
                          input_names = ['input_1'],   # the model's input names
                          output_names = ['output_1']) # the model's output names
                          # output_names = ['output']) # the model's output names
                          # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                          #               'output' : {0 : 'batch_size'}})


        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model)
        tf_model_path = onnx_model_path.replace('.onnx', '_saved_model')
        tf_rep.export_graph(tf_model_path)



        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        tflite_model_path = tf_model_path + '.tflite'
        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
    

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Export TT Stacks model')
    
    parser.add_argument('--model', help='Model path to export', type=str, required=True)
    parser.add_argument('--opset_version', help='opset_version -> check doc from torch.onnx.export', type=int, default=13)
    

    args = parser.parse_args()

    main(args)
