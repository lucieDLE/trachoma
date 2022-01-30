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
from create_stack import *


def main(args): 

    model_seg = tf.keras.models.load_model(args.seg_model)
    model_predict = tf.keras.models.load_model(args.predict_model)

    img_out = []

    if args.csv:

        with open(args.csv) as csvfile:
            df = pd.read_csv(csvfile)

            for idx, row in df.iterrows():
                img_out.append({'img': row["img"]})

    else:
        img_out.append({'img': args.img})

    pred = []
    x_a = []
    x_v = []
    x_s = []
    x_v_p = []

    for obj in img_out:


        img = sitk.ReadImage(obj["img"])  

        out_stack, seg = create_stack(img, model_seg, args)

        out_np_stack = sitk.GetArrayFromImage(out_stack)

        pred_np, x_a_np, x_v_np, x_s_np, x_v_p_np = model_predict.predict(np.array([out_np_stack]).astype(np.float32))


        pred.append(pred_np)
        x_a.append(x_a_np)
        x_v.append(x_v_np)
        x_s.append(x_s_np)
        x_v_p.append(x_v_p_np)

    pred = np.concatenate(pred)
    x_a = np.concatenate(x_a)
    x_v = np.concatenate(x_v)
    x_s = np.concatenate(x_s)
    x_v_p = np.concatenate(x_v_p)
    with open(args.out, 'wb') as f:
        pickle.dump((pred, x_a, x_v, x_s, x_v_p), f)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the image stack from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')

    in_group.add_argument('--csv', type=str, help='CSV file with images. ')
    

    parser.add_argument('--seg_model', type=str, help='Segmentation tensorflow model', default='/work/jprieto/data/remote/EGower/jprieto/trained/eyes_cropped_resampled_512_seg_train_random_rot_09072021')
    parser.add_argument('--predict_model', type=str, help='Prediction model tensorflow model', default='/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_resnet_att_17012022')
    #previous predict stacks model /work/jprieto/data/remote/EGower/jprieto/trained/stack_training_resnet_att_06012022

    parser.add_argument('--stack_size', type=int, help='Size w/h of the image stacks/frames', default=448)  
    parser.add_argument('--stack_samples', type=int, help='Stack samples', default=16)
    parser.add_argument('--out', type=str, help='Output prediction', default="out.pickle")

    args = parser.parse_args()
    main(args)