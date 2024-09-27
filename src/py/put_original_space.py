import itk
import sys
import numpy as np
import argparse
import pandas as pd
import os 

def main(args):

  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


  for img_name in os.listdir(args.in_dir):

    origin_seg = os.path.join(args.origin_seg_dir, img_name)
    updated_seg = os.path.join(args.in_dir, img_name)
    out_seg = os.path.join(args.out_dir, img_name)
    
    ImageType = itk.Image[itk.UC, 2]
                
    img_read = itk.ImageFileReader[ImageType].New(FileName=origin_seg)
    img_read.Update()
    img = img_read.GetOutput()


    img_read_l = itk.ImageFileReader[ImageType].New(FileName=updated_seg)
    img_read_l.Update()
    img_l = img_read_l.GetOutput()


    img_np = itk.GetArrayViewFromImage(img)
    img_l_np = itk.GetArrayViewFromImage(img_l)

    img_np[:,:] = 0

    origin = img_l.GetOrigin()
    size = img_l.GetLargestPossibleRegion().GetSize()

    img_np[int(origin[1]):int(origin[1] + size[1]),int(origin[0]):int(origin[0] + size[0])] = img_l_np


    writer = itk.ImageFileWriter.New(FileName=out_seg, Input=img)
    writer.UseCompressionOn()
    writer.Update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_dir', type=str, help='input dir containing corrected segmentation', required=True)
    parser.add_argument('--origin_seg_dir', type=str, help='directory containing original segmentations', required=True)
    parser.add_argument('--out_dir', type=str, help='output directory to save the segmentation', default='./corrected_seg/')

    args = parser.parse_args()
    main(args)
