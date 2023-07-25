import logging
import os
import sys
import math

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from pytorch_lightning.callbacks.quantization import wrap_qat_forward_context
from pytorch_lightning.callbacks import QuantizationAwareTraining

from nets.classification import EfficientnetV2s, MobileNetV2, MobileNetV2Stacks, TTPrediction, TTFeatures
from loaders.tt_dataset import TTDatasetSeg, TrainTransformsSeg, ExportTransformsSeg

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch

import onnx
from onnx_tf.backend import prepare

import tensorflow as tf 

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


def main_q():    

    # csv_fn = "Analysis_Set_202208/trachoma_bsl_mtss_besrat_field_seg_train_202208_eval.csv"
    # test_df = pd.read_csv(csv_fn)

    # # create a training data loader
    # test_ds = monai.data.Dataset(data=TTDatasetSeg(test_df, mount_point="./", img_column="img_path", seg_column="seg_path"), transform=ExportTransformsSeg())

    # example = test_ds[0]["img"]/255.0
    # print(example.shape)

    # model_path = "/work/jprieto/data/trachoma/train/train_patch_mobilnet_v2_torch/Analysis_Set_202208/patches_qat/model.pt"
    # model = MobileNetV2().model
    # model.load_state_dict(torch.load(model_path))
    # model.eval()


    model_path = "/work/jprieto/data/trachoma/train/train_patch_mobilnet_v2_torch/Analysis_Set_202208/patches_qat_v2/epoch=7-val_loss=0.18.ckpt"
    
    model = MobileNetV2()


    modules_to_fuse = []

    for name, module in model.named_modules():
        # print(name)
        try:
            if (isinstance(module, torch.nn.Conv2d) 
                and isinstance(model.get_submodule(replace_last(name, '.0', '.1')), torch.nn.BatchNorm2d) 
                and  isinstance(model.get_submodule(replace_last(name, '.0', '.2')), torch.nn.ReLU)):
                modules_to_fuse.append((name, replace_last(name, '.0', '.1'), replace_last(name, '.0', '.2')))
        except:
            continue

    print(modules_to_fuse)

    # QuantStub converts tensors from floating point to quantized
    model.quant = torch.quantization.QuantStub()
    # DeQuantStub converts tensors from quantized to floating point
    model.dequant = torch.quantization.DeQuantStub()
    # manually specify where tensors will be converted from quantized
    # to floating point in the quantized model
    model.__module_forward = model.forward

    model.forward = wrap_qat_forward_context(QuantizationAwareTraining(qconfig='qnnpack', observer_type="histogram", modules_to_fuse=modules_to_fuse),
        model=model, func=model.forward, trigger_condition=None
    )

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'fbgemm' for server inference
    
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')    
    
    # torch.quantization.fuse_modules(model, [], inplace=True)

    # Prepare the model for QAT. This inserts observers and fake_quants in
    # the model that will observe weight and activation tensors during calibration.
    torch.quantization.prepare_qat(model, inplace=True)

    model = model.load_from_checkpoint(model_path)


    # model = model.model
    # torch.save(model.state_dict(), "/work/jprieto/data/trachoma/train/train_patch_mobilnet_v2_torch/Analysis_Set_202208/patches_qat/epoch=60-val_loss=0.12.pt")        
    # epoch=76-val_loss=0.11.pt")
    model.eval()

    example = torch.rand(1, 3, 448, 448)
    out = model(example)
    print(out.shape)


def main():    


    model_path = "/work/jprieto/data/trachoma/train/train_patch_mobilnet_v2_torch/Analysis_Set_202208/patches_v2/epoch=12-val_loss=0.18.ckpt"
    
    model = MobileNetV2().load_from_checkpoint(model_path)
    model.eval()

    model_patches = nn.Sequential(
        model.model[0], 
        model.model[1], 
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1)
    )

    print(model_patches)

    model_features = TTFeatures(model_patches)
    model_features.eval()

    example = torch.rand(1, 448, 448, 3, dtype=torch.float32)    

    out = model_features(example)
    print(out.shape)
    
    traced_script_module = torch.jit.trace(model_features, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(model_path.replace(os.path.splitext(model_path)[1], ".ptl"))

    onnx_model_path = model_path.replace(os.path.splitext(model_path)[1], ".onnx")

    torch.onnx.export(traced_script_module,               # model being run
    example,                         # model input (or a tuple for multiple inputs)
    onnx_model_path,   # where to save the model (can be a file or file-like object)
    opset_version=12,
    do_constant_folding=True,
    export_params=True,
    input_names = ['input_1'],   # the model's input names
    output_names = ['output_1']) # the model's output names)

    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_model_path = onnx_model_path.replace('.onnx', '_saved_model')
    tf_rep.export_graph(tf_model_path)

    # tf_model = tf.saved_model.load(tf_model_path)
    # out = tf_model(input=tf.random.normal((1, 448, 448, 3)))
    # print(out)
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_model_path = tf_model_path + '.tflite'
    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


    model_path = "/work/jprieto/data/remote/EGower/jprieto/train/classification/Analysis_Set_202208/stacks_v5_mobilenet_v2/epoch=1-val_loss=0.09.ckpt"
    
    model = MobileNetV2Stacks().load_from_checkpoint(model_path)
    model.eval()


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
                      opset_version=12,
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

    

if __name__ == "__main__":
    
    main()
