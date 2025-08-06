import os
import pandas as pd
import numpy as np 

import torch
import cv2

import torchvision
import SimpleITK as sitk

from  PIL  import  Image

def compute_patch_size(seg, pad):
  seg = torch.Tensor(seg)
  shape = seg.shape

  ij = torch.argwhere(seg.squeeze() ==3 )
  bb = torch.tensor([0, 0, 0, 0]) # xmin, ymin, xmax, ymax

  bb[0] = torch.clip(torch.min(ij[:,1]) - pad, 0, shape[1])
  bb[1] = torch.clip(torch.min(ij[:,0]) - pad, 0, shape[0])
  bb[2] = torch.clip(torch.max(ij[:,1]) + pad, 0, shape[1])
  bb[3] = torch.clip(torch.max(ij[:,0]) + pad, 0, shape[0])

  patch_w = (bb[2] - bb[0])/10

  patch_size = min(int(patch_w), 256)
  return patch_size, bb

def compute_patch_location(seg, pad=29, step=20):
  y, x = np.where(seg == 3)	

  z = np.polyfit(x, y, 3)
  poly = np.poly1d(z)

  x_axis = np.arange(pad, seg.shape[-1]-pad , step=step)
  y_axis = poly(x_axis)
  return x_axis, y_axis

def extract_patches(image, x_list, y_list, patch_sz):
  patches = []
  for (x_center, y_center) in zip(x_list,y_list):
    y_center=int(y_center)
    y_center = max(patch_sz, y_center)
    y_center = min(image.shape[0] - patch_sz, y_center)

    x_center = max(patch_sz, x_center)
    x_center = min(image.shape[1] - patch_sz, x_center)
    
    patch = image[y_center-patch_sz:y_center+patch_sz, x_center-patch_sz:x_center+patch_sz]
    patches.append(patch)
  return np.stack(patches, axis=0)

def isPointInBoundingBox(point, bb):
    x1, y1, x2, y2 = bb
    x, y = point
    if (x1 <= x <= x2):
      if (y1 <= y <= y2):
        return True
    return False



img_dir = '/CMF/data/lumargot/trachoma/PoPP_Data/mtss/img'
seg_dir = '/CMF/data/lumargot/trachoma/PoPP_Data/mtss/seg'
csv_dir = '/CMF/data/lumargot/trachoma/PoPP_Data/mtss/old_patches'
out_dir = '/CMF/data/lumargot/trachoma/PoPP_Data/mtss/256_patches'



csv_list = os.listdir(csv_dir)
num_subject = len(csv_list)
map = { 'Healthy':0, 'ECA':1, 'Entropion':2, 'Gap':3, 'overcorrection':4, 'Short Incision':5,  'Fleshy':6}

l_patch, l_cid, l_x, l_y, l_label, l_class, l_img = [], [], [], [], [],[],[]

for index_csv in range(num_subject):
  csv_name = csv_list[index_csv]
  csv_file = os.path.join(csv_dir, csv_name)

  df = pd.read_csv(csv_file)
  num_patches = len(df)
  subject_name = os.path.splitext(csv_name)[0]
  subject_path = os.path.join(img_dir, subject_name+'.jpg')
  seg_path =  os.path.join(seg_dir, subject_name+'.nrrd')

  image  = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(subject_path)))
  seg  = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)))
  H,W = image.shape[:2]

  patch_sz, bb = compute_patch_size(seg, pad=1)
  for idx, row in df.iterrows():
    x,y, label = row['x'], row['y'], row['label']
    if '/' in label:
      label = label.replace('/', '_')
    if label == 'Wavy_ECA':
      label = 'ECA'

    if label in map.keys():

      xmin = int(max((x-patch_sz), 0))
      ymin = int(max((y-5*patch_sz/3), 0))

      xmax = int(min((x+patch_sz), W))
      ymax = int(min((y+patch_sz/3), H))

      img = image[ymin:ymax, xmin:xmax]
      img_tmp = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
      
      name = f"patch_{subject_name}_{x}x_{y}y_class_{label}.jpg"

      l_img.append(os.path.join('img', subject_name +'.jpg'))
      l_cid.append(subject_name.split('_')[0])
      l_x.append(x)
      l_y.append(y)
      l_label.append(label)
      l_class.append(int(map[label]))
      l_patch.append(os.path.join('patches', name))

      pi_up_img = Image.fromarray(img_tmp)
      pi_up_img.save(os.path.join(out_dir, name))

df_out = pd.DataFrame(data={'patch':l_patch,
                            'cid':l_cid,
                            'x_patch':l_x,
                            'y_patch':l_y,
                            'label':l_label,
                            'class':l_class,
                            'filename':l_img
})
df_out.to_csv('/CMF/data/lumargot/trachoma/PoPP_Data/annotations_set_062025.csv')