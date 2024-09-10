import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import cv2
import SimpleITK as sitk
import argparse

def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

def main(args):

    seg_paths = []
    i = 1


    df = pd.read_csv(args.in_csv)

    for idx, row in df.iterrows():

        full_img = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(row['img_path'])).copy())

        seg = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(row['crop_seg'])).copy())
        box= converter(row['box'])
        xmin, ymin, xmax, ymax  = box.astype(int)

        full_seg = np.zeros_like(full_img[:,:,0])
        print(full_seg.shape, seg.shape)
        print(ymin, ymax, xmin, xmax)
        full_seg[ymin:ymax, xmin:xmax] = seg
        full_seg = sitk.GetImageFromArray(full_seg, isVector=True)

        full_seg_path = row['crop_seg'].replace('crop_seg', 'seg')
        seg_paths.append(full_seg_path)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(full_seg_path)
        writer.UseCompressionOn()
        writer.Execute(full_seg)


    df['seg'] = seg_paths
    df.to_csv(args.out_csv)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_csv', type=str, help='input csv file', required=True)
    parser.add_argument('--out_csv', type=str, help='output csv file', default="data.csv")

    args = parser.parse_args()
    main(args)

    ## to do; all column as default args