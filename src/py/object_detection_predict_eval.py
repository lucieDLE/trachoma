from tqdm import tqdm
import pandas as pd
import numpy as np 

import torch
import nrrd
import matplotlib.pyplot as plt
import json
import itertools
from matplotlib.patches import Rectangle
from sklearn.metrics import classification_report
import pdb
from torchvision.ops import nms

import matplotlib.patches as mpatches
import numpy as np
import cv2
from collections import defaultdict
import os 
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report, confusion_matrix

from utils import remove_labels
from visualization import *
from evaluation import *
from utils import *
from nets.segmentation import FasterTTRCNN
from loaders.tt_dataset import TTDatasetBX,TTDataModuleBX, BBXImageTrainTransform, BBXImageEvalTransform, BBXImageTestTransform

from sklearn.metrics import classification_report, confusion_matrix
from nets.segmentation import TTRoidHead, FasterTTRCNN, TTRPN
from collections import OrderedDict
from torchvision.ops import nms


from visualization import *
from evaluation import *

from tqdm import tqdm
import json

def eye_level_outcome(preds,img_shape):
  unique_labels, counts = np.unique(preds['labels'], return_counts=True)
  unique_labels = unique_labels[counts >=3]
  counts = counts[counts>=3]

  outcome = -1 

  if len(unique_labels) == 3:
    labels, ct = unique_labels[1:], counts[1:]
    max_labels = labels[ np.argwhere(ct == np.amax(ct))][:,0]
    if np.std(ct) >=1:
      outcome = max_labels.item()
    else:
      outcome = -1

  elif len(unique_labels) == 2:
    if 1 in unique_labels:
      outcome = max(unique_labels)
    else: 
      if np.std(counts) >=1:
        max_labels = unique_labels[ np.argwhere(counts == np.amax(counts))][:,0]
        outcome = max(max_labels).item()
      else:
        outcome = -1
  elif len(unique_labels) == 1:
      outcome = unique_labels.item()
  else:
    outcome =-1
  
  return outcome 
def x_iou(boxA, boxB):
    """Compute IoU only along the x-axis."""
    x1_A, y1_A, x2_A, y2_A = boxA
    x1_B, y1_B, x2_B, y2_B = boxB
    
    if (y2_A >= y1_B or y2_B >= y1_A):
        
        inter = max(0, min(x2_A, x2_B) - max(x1_A, x1_B))
        union = (x2_A - x1_A) + (x2_B - x1_B) - inter
        return inter / union if union > 0 else 0
    return 0

def y_overlap(boxA, boxB):
    """Check if boxes overlap in the y-axis."""
    return not (boxA[3] <= boxB[1] or boxB[3] <= boxA[1])
    
def custom_x_nms(preds, iou_thresh=0.8):
    """ Apply NMS based only on x-axis IoU. """
    boxes = preds['boxes']
    if 'scores' in preds.keys():
        scores = preds['scores']
    else:
        scores = torch.ones_like(preds['labels'])

    idxs = np.argsort(-scores)  # sort descending by score
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        rest = idxs[1:]
        
        ious = np.array([x_iou(boxes[current], boxes[i]) for i in rest])
        idxs = rest[ious <= iou_thresh]

    return torch.stack(keep)

def process_predictions(preds):
  keep = preds['scores'] > 0.1
  preds = filter_targets_indices(preds, keep, detach=True)
  if len(preds['scores']) >= 1:
    pred_indices = custom_x_nms(preds, iou_thresh=0.4)
    preds = filter_targets_indices(preds, pred_indices, detach=False)
  if len(preds['scores']) >= 1:
    preds = fill_empty_patches(preds)
  if len(preds['scores']) >= 1:
    preds = replace_sandwiched_labels(preds)
  return preds


mnt_ckpt = '/CMF/data/lumargot/trachoma/output/backtoold'

all_checkpoints = [



'5fold_v6/fold0/epoch=10-val_loss=0.508.ckpt', 
'5fold_v6/fold0/epoch=6-val_loss=0.512.ckpt', 
'5fold_v6/fold0/last-v1.ckpt',

'5fold_v6/fold1/epoch=10-val_loss=0.499.ckpt', 
'5fold_v6/fold1/epoch=14-val_loss=0.495.ckpt', 
'5fold_v6/fold1/last-v1.ckpt', 

'5fold_v6/fold2/epoch=12-val_loss=0.610.ckpt', 
'5fold_v6/fold2/epoch=5-val_loss=0.606.ckpt', 
'5fold_v6/fold2/last-v1.ckpt', 

'5fold_v6/fold3/epoch=11-val_loss=0.632.ckpt', 
'5fold_v6/fold3/epoch=13-val_loss=0.636.ckpt', 
'5fold_v6/fold3/last-v1.ckpt', 

'5fold_v6/fold4/epoch=8-val_loss=0.525.ckpt', 
'5fold_v6/fold4/epoch=9-val_loss=0.515.ckpt', 
'5fold_v6/fold4/last-v1.ckpt', 


'5fold_df/fold0/epoch=11-val_loss=1.053.ckpt',
'5fold_df/fold0/epoch=6-val_loss=1.059.ckpt',
'5fold_df/fold0/last.ckpt',


'5fold_df/fold1/epoch=14-val_loss=0.909.ckpt',
'5fold_df/fold1/epoch=9-val_loss=0.902.ckpt',
'5fold_df/fold1/last.ckpt',

'5fold_df/fold2/epoch=11-val_loss=0.969.ckpt',
'5fold_df/fold2/epoch=8-val_loss=0.976.ckpt',
'5fold_df/fold2/last.ckpt',

'5fold_df/fold3/epoch=9-val_loss=0.922.ckpt',
'5fold_df/fold3/epoch=9-val_loss=0.935.ckpt',
'5fold_df/fold3/last.ckpt',


'5fold_df/fold4/epoch=7-val_loss=0.959.ckpt',
'5fold_df/fold4/epoch=8-val_loss=0.984.ckpt',
'5fold_df/fold4/last.ckpt',

]

out_dir = '/CMF/data/lumargot/trachoma/output/backtoold/5fold_v6/test/fold2'


# === old === #
mount_point = "/CMF/data/lumargot/trachoma/"

df_train = pd.read_csv('/CMF/data/lumargot/trachoma/csv_updated/mtss_pret_combined_train_fold2_train_train.csv')
df_val = pd.read_csv('/CMF/data/lumargot/trachoma/csv_updated/mtss_pret_combined_train_fold2_train_test.csv')
df_test = pd.read_csv('/CMF/data/lumargot/trachoma/csv_updated/mtss_pret_combined_train_fold2_test.csv')

# df_test = pd.read_csv('/CMF/data/lumargot/trachoma/csv_updated/mtss_pret_combined_test.csv')
df_test = df_test.drop_duplicates(subset=['x_patch', 'y_patch', 'filename'])

concat_labels=['overcorrection', 'ECA', 'Gap', 'Fleshy']
drop_labels = ['Short Incision', 'Reject']

img_column= "filename" 
class_column = 'class'
label_column = 'label'

map ={ 1:'Healthy', 2:'Entropion', 3:'Overcorrection'}


df_test = remove_labels(df_test, class_column, label_column, drop_labels=drop_labels, concat_labels=concat_labels)
df_train = remove_labels(df_train, class_column, label_column, drop_labels=drop_labels, concat_labels=concat_labels)
df_val = remove_labels(df_val, class_column, label_column, drop_labels=drop_labels, concat_labels=concat_labels)

df_test = df_test.loc[df_test['dataset'] == 'PoPP_Data']

ttdata = TTDataModuleBX(df_train, df_val, df_test, batch_size=1, num_workers=1, img_column='filename',severity_column='sev', 
                        mount_point=mount_point, class_column= class_column,
                        train_transform=BBXImageTestTransform(), 
                        valid_transform=BBXImageTestTransform(), 
                        test_transform=BBXImageTestTransform(height=666, width=1333))
ttdata.setup()
dataload = ttdata.test_dataloader()
ds = ttdata.test_ds

for ckpt_name in all_checkpoints:
    ckpt = os.path.join(mnt_ckpt, ckpt_name)
    model = FasterTTRCNN.load_from_checkpoint(ckpt, strict=True)
    model.eval()
    model.cuda()


    num_preds,num_fps,num_fns = 0, 0, 0
    l_ious, l_distances = [], []
    l_distances = []
    gt, pred = [], []
    data_out = {}
    lcid,leye = [], []
    datasets = []
    gt_eye, pred_eye = [], []
    probs= []
    for idx, batch in enumerate(tqdm(dataload)):
    
        targets = batch
        imgs = targets.pop('img', None)
        outs = model(imgs, mode='test')
        out_img = outs[0]

        # remove overlapping boxes with iou > 0.7  
        ### -- gt -- ###
        gt_indices = nms(targets['boxes'][0], torch.ones_like(targets['boxes'][0,:,0]), iou_threshold=0.9) ## iou as args
        filename = ds.data.df_subject.iloc[idx]['filename']
        targets['boxes'] = targets['boxes'][0,gt_indices].cpu().detach()
        targets['labels'] = targets['labels'][0,gt_indices].cpu().detach()  

        ### -- preds -- ###
        # eyelid_seg = select_eyelid_seg(targets['mask'][0])
        # preds = filter_indices_on_segmentation_mask(eyelid_seg, out_img, overlap_threshold=0.5)
        preds = process_predictions(out_img)


        ## box-level evaluation
        n_p, n_fp, n_fn, i, d, gt_ix, pred_idx = evaluate_with_fp_fn(targets['boxes'], preds['boxes'])

        gt.append(targets['labels'][gt_ix])
        pred.append(preds['labels'][pred_idx])
        probs.append(preds['scores'][pred_idx])

        num_preds += n_p
        num_fps += n_fp
        num_fns += n_fn
        l_ious.append(torch.tensor(i).reshape(-1))
        l_distances.append(torch.tensor(d))


    ious = torch.cat(l_ious, dim=0)
    dist = torch.cat(l_distances, dim=0)
    pred = np.concatenate(pred)
    gt = np.concatenate(gt)

    total_detections = num_preds + num_fns + num_fps
    detect_stats = {'Matched Prediction': num_preds,
                    'Ratio match prediction': 100*num_preds/total_detections, 
                    'False Positives':num_fps,
                    'Ratio FP': 100*num_fps/total_detections, 
                    'False Negatives':num_fns,
                    'Ration FN': 100*num_fns/total_detections, 
                    'Mean IoU': ious.mean().item(),
                    }

    name = os.path.splitext(ckpt)[0]

    json_file = os.path.join(out_dir, name +  '_mtss_box_detect_report.json')
    with open(json_file, "w") as f:
        json.dump(detect_stats, f, indent=2) # indent for pretty printing


    df_pret = pd.DataFrame(data={'gt':gt, 'pred':pred})
    df_pret.to_csv(os.path.join(out_dir, name + '_mtss_box-level_prediction.csv'))

    report = classification_report(df_pret['gt'], df_pret['pred'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(out_dir, name +  '_mtss_box-level_report.csv'))


    fig = plt.figure(figsize=(16,6))
    plt.subplot(121)
    cnf_matrix = confusion_matrix(df_pret['gt'], df_pret['pred'])
    plot_confusion_matrix(cnf_matrix, classes=['Healthy', 'Undercorrection', 'Overcorrection'], title='confusion matrix')
    plt.subplot(122)
    cm = plot_confusion_matrix(cnf_matrix, classes=['Healthy', 'Undercorrection', 'Overcorrection'], normalize=True, title='confusion matrix - normalized')
    plt.savefig(os.path.join(out_dir, name + '_mtss_box-level_cm.png'),dpi=200)