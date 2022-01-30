import SimpleITK as sitk
import numpy as np
import argparse
from collections import namedtuple

import tensorflow as tf

import resample
import poly_fit as pf
import os
import pickle
import pandas as pd

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

def main(args): 

    model_seg = tf.keras.models.load_model(args.seg_model)

    img_out = []

    if args.csv:

        with open(args.csv) as csvfile:
            df = pd.read_csv(csvfile)

            for idx, row in df.iterrows():
                img = row["img"]

                if args.csv_root:
                    img = img.replace(args.csv_root, '')

                out = os.path.normpath(args.out + img).replace(".jpg", ".nrrd")

                out_dir = os.path.dirname(out)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                out_seg = None

                if args.out_seg:

                    out_seg = os.path.normpath(args.out_seg + img).replace(".jpg", ".nrrd")                    

                    out_seg_dir = os.path.dirname(out_seg)

                    if not os.path.exists(out_seg_dir):
                        os.makedirs(out_seg_dir)
                
                if args.ow or (not os.path.exists(out) and not os.path.exists(out_seg)):
                    img_out.append({'img': row["img"], 'out': out, 'out_seg': out_seg})

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
            print(bcolors.FAIL, e, bcolors.ENDC)
    

def create_stack(img, model_seg, args):

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

    seg_resampled_np = model_seg.predict(np.array([img_resampled_np]))[0]

    seg_resampled = sitk.GetImageFromArray(seg_resampled_np.astype(np.ubyte), isVector=True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the image stack from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')

    in_group.add_argument('--csv', type=str, help='CSV file with images. ')
    parser.add_argument('--csv_root', type=str, help='Root path to replace for output', default=None)
    
    parser.add_argument('--seg_model', type=str, help='Segmentation tensorflow model', default='/work/jprieto/data/remote/EGower/jprieto/trained/eyes_cropped_resampled_512_seg_train_random_rot_09072021')

    parser.add_argument('--stack_size', type=int, help='Size w/h of the image stacks/frames', default=448)  
    parser.add_argument('--stack_samples', type=int, help='Stack samples', default=16)  

    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--out_seg', type=str, help='Output seg dir', default=None) 
    output_group.add_argument('--out', type=str, help='Output stacks dir', default="out/")    
    output_group.add_argument('--ow', type=bool, help='Overwrite outputs', default=False)    

    args = parser.parse_args()
    main(args)