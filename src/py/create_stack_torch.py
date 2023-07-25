import SimpleITK as sitk
import numpy as np
import argparse
from collections import namedtuple

import torch
import monai

from monai.transforms import (
    AsChannelFirst,
    ScaleIntensity,
    ToTensor,
    ToNumpy,
    AsChannelLast, 
    Lambda,
    Compose
)

import resample
import poly_fit as pf
import os
import sys
import pickle
import pandas as pd

import tensorflow as tf

class bcolors:
    HEADER = '\033[95m'
    OK = '\033[94m'
    INFO = '\033[96m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_stack(img, model_seg, args):

    device = torch.device("cuda:0")

    transforms_in = Compose(
        [
            AsChannelFirst(),
            ScaleIntensity(),
            ToTensor(dtype=torch.float32),
            Lambda(func=lambda x: torch.unsqueeze(x, dim=0))
        ]
    )

    transforms_out = Compose(
        [
            AsChannelLast(channel_dim=1),
            Lambda(func=lambda x: torch.argmax(x, dim=-1, keepdim=True)),
            Lambda(func=lambda x: torch.squeeze(x, dim=0)),
            ToNumpy(dtype=np.ubyte)
        ]
    )

    resample_obj = {}
    resample_obj["size"] = [512, 512]
    resample_obj["fit_spacing"] = True
    resample_obj["iso_spacing"] = True
    resample_obj["pixel_dimension"] = 3
    resample_obj["center"] = False  
    resample_obj["linear"] = True
    resample_obj["spacing"] = None
    resample_obj["origin"] = None

    resample_args = namedtuple("resample_args", resample_obj.keys())(*resample_obj.values())

    img_resampled = resample.resample_fn(img, resample_args)

    img_resampled_np = sitk.GetArrayFromImage(img_resampled)

    img_resampled_np = transforms_in(img_resampled_np).to(device)
    seg_resampled_np = model_seg(img_resampled_np)
    seg_resampled_np = transforms_out(seg_resampled_np)

    seg_resampled = sitk.GetImageFromArray(seg_resampled_np, isVector=True)
    seg_resampled.SetSpacing(img_resampled.GetSpacing())
    seg_resampled.SetOrigin(img_resampled.GetOrigin())
    
    resample_obj = {}
    resample_obj["size"] = img.GetSize()
    resample_obj["fit_spacing"] = False
    resample_obj["iso_spacing"] = False
    resample_obj["image_dimension"] = 2
    resample_obj["pixel_dimension"] = 1
    resample_obj["center"] = False  
    resample_obj["linear"] = False
    resample_obj["spacing"] = img.GetSpacing()
    resample_obj["origin"] = img.GetOrigin()

    resample_args = namedtuple("resample_args", resample_obj.keys())(*resample_obj.values())
    

    seg = resample.resample_fn(seg_resampled, resample_args)
    
    seg_np = sitk.GetArrayFromImage(seg)
    img_np = sitk.GetArrayFromImage(img)    

    out_np_stack = pf.poly_fit(img_np, seg_np, 3, args.stack_size, args.stack_samples)

    out_stack = sitk.GetImageFromArray(out_np_stack, isVector=True)

    return out_stack, seg

def main(args): 

    device = torch.device("cuda:0")

    model_seg = monai.networks.nets.UNet(spatial_dims=2, in_channels=3, out_channels=4, channels=(16, 32, 64, 128, 256, 512, 1024), strides=(2, 2, 2, 2, 2, 2), num_res_units=4)    
    model_seg.load_state_dict(torch.load(args.seg_model, map_location=device))
    model_seg.to(device)

    img_out = []

    # model_predict = None

    # if args.predict_model:
    #     model_predict = tf.keras.models.load_model(args.predict_model)

    if args.csv:

        with open(args.csv) as csvfile:
            df = pd.read_csv(csvfile)

            for idx, row in df.iterrows():
                img = row[args.img_column]

                if args.csv_root:
                    img = img.replace(args.csv_root, '')

                out = os.path.normpath(os.path.join(args.out, img)).replace(".jpg", ".nrrd")

                out_dir = os.path.dirname(out)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                out_seg = None

                if args.out_seg:

                    out_seg = os.path.normpath(os.path.join(args.out_seg, img)).replace(".jpg", ".nrrd")                    

                    out_seg_dir = os.path.dirname(out_seg)

                    if not os.path.exists(out_seg_dir):
                        os.makedirs(out_seg_dir)
                
                if args.ow or (not os.path.exists(out) or not os.path.exists(out_seg)):
                    img_out.append({'img': row[args.img_column], 'out': out, 'out_seg': out_seg})

    else:
        img_out.append({'img': args.img, 'out': args.out})

    pred = []
    x_a = []
    x_v = []
    x_s = []
    x_v_p = []

    for obj in img_out:

        try:
            print(bcolors.INFO, "Reading:", obj["img"], bcolors.ENDC)
            img = sitk.ReadImage(obj["img"])  

            out_stack, seg = create_stack(img, model_seg, args)

            if obj["out_seg"] is not None:
                print("Writing:", obj["out_seg"])
                writer = sitk.ImageFileWriter()
                writer.SetFileName(obj["out_seg"])
                writer.UseCompressionOn()
                writer.Execute(seg)

            print(bcolors.SUCCESS, "Writing:", obj["out"], bcolors.ENDC)
            writer = sitk.ImageFileWriter()
            writer.SetFileName(obj["out"])
            writer.UseCompressionOn()
            writer.Execute(out_stack)
        except Exception as e:
            print(bcolors.FAIL, e, bcolors.ENDC, file=sys.stderr)


    #     if model_predict:
    #         try:
    #             out_np_stack = sitk.GetArrayFromImage(out_stack)
    #             pred_np, x_a_np, x_v_np, x_s_np, x_v_p_np = model_predict.predict(np.array([out_np_stack]).astype(np.float32))

    #             pred.append(pred_np)
    #             x_a.append(x_a_np)
    #             x_v.append(x_v_np)
    #             x_s.append(x_s_np)
    #             x_v_p.append(x_v_p_np)
    #         except Exception as e:
    #             print(bcolors.FAIL, e, bcolors.ENDC, file=sys.stderr)

    # if model_predict:
    #     pred = np.concatenate(pred)
    #     x_a = np.concatenate(x_a)
    #     x_v = np.concatenate(x_v)
    #     x_s = np.concatenate(x_s)
    #     x_v_p = np.concatenate(x_v_p)

    #     out_name = os.path.normpath(args.out) + "_predict.pickle"
    #     with open(args.out, 'wb') as f:
    #         pickle.dump((pred, x_a, x_v, x_s, x_v_p), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the image stack from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')

    in_group.add_argument('--csv', type=str, help='CSV file with images. Uses column name "image"')
    parser.add_argument('--csv_root', type=str, help='Root path to replace for output', default=None)
    parser.add_argument('--img_column', type=str, help='Name of column in csv file', default="image")
    
    parser.add_argument('--seg_model', type=str, help='Segmentation torch model', default='/work/jprieto/data/remote/EGower/jprieto/train/torch_unet_train_01252022/model.pt')

    # parser.add_argument('--predict_model', type=str, help='Segmentation torch model', default='/work/jprieto/data/remote/EGower/jprieto/trained/train_stack_efficientv2s_01042022_saved_model')

    parser.add_argument('--stack_size', type=int, help='Size w/h of the image stacks/frames', default=768)  
    parser.add_argument('--stack_samples', type=int, help='Stack samples', default=16)  

    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--out_seg', type=str, help='Output seg dir', default=None) 
    output_group.add_argument('--out', type=str, help='Output stacks dir', default="out/")    
    output_group.add_argument('--ow', type=bool, help='Overwrite outputs', default=False)    

    args = parser.parse_args()
    main(args)