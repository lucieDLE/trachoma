import pdb
import os 
import pickle
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import cv2
import torch
import SimpleITK as sitk

from  PIL  import  Image
import requests
from io import BytesIO

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def compute_patch_size(seg):
  seg = torch.Tensor(seg)
  shape = seg.shape

  ij = torch.argwhere(seg.squeeze() !=0.)
  pad = 5
  bb = torch.tensor([0, 0, 0, 0]) # xmin, ymin, xmax, ymax

  bb[0] = torch.clip(torch.min(ij[:,1]) - pad, 0, shape[1])
  bb[1] = torch.clip(torch.min(ij[:,0]) - pad, 0, shape[0])
  bb[2] = torch.clip(torch.max(ij[:,1]) + pad, 0, shape[1])
  bb[3] = torch.clip(torch.max(ij[:,0]) + pad, 0, shape[0])

  seg_cropped = seg[ bb[1]:bb[3], bb[0]:bb[2]]

  patch_h = seg_cropped.shape[0]/10
  patch_w = seg_cropped.shape[1]/10

  patch_size = min(int(patch_w), 64)
  return seg_cropped.numpy(), patch_size





directory = '/CMF/data/lumargot/trachoma/patches/all'
out_dir = '/CMF/data/lumargot/trachoma/patches/esgran_patch/'
img_dir = '/CMF/data/lumargot/trachoma/B images one eye/img'
seg_dir = '/CMF/data/lumargot/trachoma/B images one eye/seg'
weight_path = '/CMF/data/lumargot/trachoma/weights/RealESRGAN_x4plus.pth'

SCALE=4
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=SCALE)
upsampler = RealESRGANer(scale=SCALE, model_path=weight_path, dni_weight=None, model=model)


csv_list = os.listdir(directory)
num_subject = len(csv_list)
for index_csv in range(num_subject):
  csv_name = csv_list[index_csv]
  csv_file = os.path.join(directory, csv_name)



  df = pd.read_csv(csv_file)
  num_patches = len(df)
  subject_name = os.path.splitext(csv_name)[0]
  subject_path = os.path.join(img_dir, subject_name+'.jpg')
  seg_path =  os.path.join(seg_dir, subject_name+'.nrrd')

  image  = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(subject_path)))
  seg  = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)))
  H,W = image.shape[:2]
  seg_cropped, pad = compute_patch_size(seg)
  ratio = seg_cropped.shape[0]/seg_cropped.shape[1]



  H,W = image.shape[:2]
  print(f"extracting {num_patches} patches of subject {subject_name} --> {index_csv} / {num_subject}")
  print()
  for idx,row in df.iterrows():
    x,y,label = row['x'], row['y'], row['label']
    if '/' in label:
      label = label.replace('/', '_')
    xmin, xmax = max(0,x-pad), min(x+pad, W)
    ymin, ymax = max(0,y-pad), min(y+pad, H)

    img = image[ymin:ymax, xmin:xmax]
    if img.shape[0] < 128:
      img_tmp = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    else:
      img_tmp = img

    upscaled_image, _ = upsampler.enhance(img_tmp)
    
    name = f"patch_{subject_name}_{x}x_{y}y_class_{label}.png"

    pi_up_img = Image.fromarray(upscaled_image)
    pi_up_img.save(os.path.join(out_dir, name))
