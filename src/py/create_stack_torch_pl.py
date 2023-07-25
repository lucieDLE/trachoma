import SimpleITK as sitk
import numpy as np
import argparse
from collections import namedtuple

import torch

from nets.segmentation import TTUNet
from loaders.tt_dataset import InTransformsSeg, OutTransformsSeg

import resample
import poly_fit as pf
import os
import sys
import pickle
import pandas as pd

import pickle

from sklearn.metrics import classification_report
from nets.classification import EfficientnetV2sStacks

import glob

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

    transforms_in = InTransformsSeg()

    transforms_out = OutTransformsSeg()

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

    print(bcolors.INFO, "Starting polyfit...", bcolors.ENDC)

    out_np_stack = pf.poly_fit(img_np, seg_np, 3, args.stack_size, args.stack_samples)

    out_stack = sitk.GetImageFromArray(out_np_stack, isVector=True)

    return out_stack, seg

# def create_stack_inferer(img, model_seg, args):

#     device = torch.device("cuda:0")

#     transforms_in = InTransformsSeg()

#     transforms_out = OutTransformsSeg()

#     img_np = sitk.GetArrayFromImage(img)
    
#     img_t = transforms_in(img_np).to(device)    


#     inferer = SlidingWindowInferer(roi_size=[512, 512], device=device, progress=True, overlap=0.1)

#     with torch.no_grad():
#         seg_t = inferer(inputs=img_t, network=model_seg.model)
#         print(seg_t.shape)
#         seg_t = torch.argmax(seg_t, dim=1, keepdim=True)

#     seg_np = transforms_out(seg_t)

#     seg = sitk.GetImageFromArray(seg_np, isVector=True)
#     seg.SetSpacing(img.GetSpacing())
#     seg.SetOrigin(img.GetOrigin())


#     seg_np = sitk.GetArrayFromImage(seg)
#     img_np = sitk.GetArrayFromImage(img)    

#     print(bcolors.INFO, "Starting polyfit...", bcolors.ENDC)

#     out_np_stack = pf.poly_fit(img_np, seg_np, 3, args.stack_size, args.stack_samples)

#     out_stack = sitk.GetImageFromArray(out_np_stack, isVector=True)

#     return out_stack, seg


def main(args): 

    device = torch.device("cuda:0")

    model_seg = TTUNet.load_from_checkpoint(args.seg_model)
    model_seg.eval()
    model_seg.cuda()

    img_out = []

    model_predict = None
    if args.predict_model:
        model_predict = EfficientnetV2sStacks.load_from_checkpoint(args.predict_model)

    if args.dir:
        images = []
        for file_path in glob.glob(os.path.join(args.dir,'**', '*.jpg', recursive=True):
            images.append(file_path)

        df = pd.DataFrame({args.img_column: images})

    if args.csv or args.dir:
        
        if args.csv:
            df = pd.read_csv(args.csv)

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
                    
            img_out.append({'img': row[args.img_column], 'out': out, 'out_seg': out_seg})

        df_out = pd.DataFrame(img_out)

        if len(df_out) == 0:
            print(bcolors.INFO, "All images have been processed!", bcolors.ENDC)
            quit()
        df['seg'] = df_out['out_seg']
        df['stack'] = df_out['out']

        csv_split_ext = os.path.splitext(args.csv)
        out_csv = csv_split_ext[0] + "_seg_stack" + csv_split_ext[1]
        df.to_csv(out_csv, index=False)
        
    else:
        img_out.append({'img': args.img, 'out': args.out})

    probs = []
    features = []
    scores = []
    features_v = []
    features_v_p = []

    for obj in img_out:

        if args.ow or not os.path.exists(obj["out"]):

            try:
                print(bcolors.INFO, "Reading:", obj["img"], bcolors.ENDC)
                img = sitk.ReadImage(obj["img"])  

                out_stack, seg = create_stack(img, model_seg, args)

                if obj["out_seg"] is not None:
                    print(bcolors.SUCCESS, "Writing:", obj["out_seg"], bcolors.ENDC)
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

        elif model_predict is not None:
            out_stack =  sitk.ReadImage(obj["out"])
            
        if model_predict:
            out_stack = sitk.GetArrayFromImage(out_stack)
            out_stack = torch.tensor(out_stack, dtype=torch.float32)
            out_stack = out_stack.permute((0, 3, 1, 2))
            out_stack = out_stack/255.0
            out_stack = out_stack.unsqueeze(dim=0)
            
            x, x_a, x_s, x_v, x_v_p = model_predict(out_stack)

            probs.append(x)       
            features.append(x_a)
            scores.append(x_s)
            features_v.append(x_v)
            features_v_p.append(x_v_p)

    if model_predict:
        probs = torch.cat(probs, dim=0)
        predictions = torch.argmax(probs, dim=1)

        features = torch.cat(features, dim=0).cpu().numpy()
        scores = torch.cat(scores, dim=0).cpu().numpy()
        features_v = torch.cat(features_v, dim=0).cpu().numpy()
        features_v_p = torch.cat(features_v_p, dim=0).cpu().numpy()

        probs = probs.cpu().numpy()
        df["pred"] = predictions.cpu().numpy()

        if args.class_column:
            print(classification_report(df[args.class_column], df["pred"]))

        df.to_csv(args.csv.replace(".csv", "_prediction.csv"), index=False)
        
        pickle.dump((probs, features, scores, features_v, features_v_p), open(args.csv.replace(".csv", "_features.pickle"), 'wb'))
            




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the image stack from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')
    in_group.add_argument('--csv', type=str, help='CSV file with images. Uses column name "image"')
    in_group.add_argument('--dir', type=str, help='Directory with jpg images')
    parser.add_argument('--csv_root', type=str, help='Root path to replace for output', default=None)
    parser.add_argument('--img_column', type=str, help='Name of column in csv file', default="image")
    parser.add_argument('--class_column', type=str, help='Name of class column in csv file', default=None)
    
    parser.add_argument('--seg_model', type=str, help='Segmentation torch model', default='/work/jprieto/data/remote/EGower/jprieto/train/Analysis_Set_202208/segmentation_unet/v3/epoch=490-val_loss=0.07.ckpt')

    parser.add_argument('--predict_model', type=str, help='Stack predict model', default='')

    parser.add_argument('--stack_size', type=int, help='Size w/h of the image stacks/frames', default=768)  
    parser.add_argument('--stack_samples', type=int, help='Stack samples', default=16)  

    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--out_seg', type=str, help='Output seg dir', default=None) 
    output_group.add_argument('--out', type=str, help='Output stacks dir', default="out/")    
    output_group.add_argument('--ow', type=bool, help='Overwrite outputs', default=False)    

    args = parser.parse_args()
    main(args)