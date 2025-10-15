import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import itertools
from utils import filter_targets_indices
import torch
from visualization import replace_sandwiched_labels, fill_empty_patches

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
      plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto',vmin=0, vmax=1)
    else:
      print('Confusion matrix, without normalization')
      plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = .5 if normalize else np.sum(cm)/4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    return cm

def compute_bbx_iou(gt_box, pred_box):
    """
    Compute IoU for two bounding boxes.
    Each box should be [x1, y1, x2, y2]
    """
    x1 = max(gt_box[0], pred_box[0])
    y1 = max(gt_box[1], pred_box[1])
  
    x2 = min(gt_box[2], pred_box[2])
    y2 = min(gt_box[3], pred_box[3])

    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    gt_boxArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    pred_boxArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    iou = interArea / float(gt_boxArea + pred_boxArea - interArea)
    return iou

def match_boxes(gt_boxes, pred_boxes):
    """Match predicted boxes to ground truth boxes using the Hungarian algorithm."""
    
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], []

    # Compute cost matrix (distance between centers)
    cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

    for i, (x1_gt, y1_gt, x2_gt, y2_gt) in enumerate(gt_boxes):
        center_gt = np.array([(x1_gt + x2_gt) / 2, (y1_gt + y2_gt) / 2])
        
        for j, (x1_pred, y1_pred, x2_pred, y2_pred) in enumerate(pred_boxes):
            center_pred = np.array([(x1_pred + x2_pred) / 2, (y1_pred + y2_pred) / 2])
            cost_matrix[i, j] = np.linalg.norm(center_gt - center_pred)  # Euclidean distance

    # Solve assignment problem (minimize distance)
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    return gt_indices, pred_indices

def evaluate_with_fp_fn(gt_boxes, pred_boxes):
    gt_indices, pred_indices = match_boxes(gt_boxes, pred_boxes)

    distances = []
    ious = []
    num_pred =0
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        num_pred+=1
        x1_gt, y1_gt, x2_gt, y2_gt = gt_boxes[gt_idx]
        x1_pred, y1_pred, x2_pred, y2_pred = pred_boxes[pred_idx]

        iou = compute_bbx_iou(gt_boxes[gt_idx], pred_boxes[pred_idx])
        ious.append(iou)
        # Compute center distance
        center_gt = np.array([(x1_gt + x2_gt) / 2, (y1_gt + y2_gt) / 2])
        center_pred = np.array([(x1_pred + x2_pred) / 2, (y1_pred + y2_pred) / 2])
        distances.append(np.linalg.norm(center_gt - center_pred))
    
    # Count false positives (unmatched predictions)
    num_false_positives = len(pred_boxes) - len(pred_indices)

    # Count false negatives (unmatched GT boxes)
    num_false_negatives = len(gt_boxes) - len(gt_indices)
    return num_pred, num_false_positives, num_false_negatives, ious, distances, gt_indices, pred_indices

def gt_eye_outcome(targets_labels):
  unique_labels = np.unique(targets_labels, return_counts=False)
  if len(unique_labels) == 2:
    if 1 in unique_labels:
      outcome = max(unique_labels)
    else: #conflict
      outcome = -1 
  elif len(unique_labels) == 3: #conflict 
    outcome = -1
  else:
    outcome = unique_labels[0]
  
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
  
def eye_level_outcome(preds,img_shape):
  unique_labels, counts = np.unique(preds['labels'], return_counts=True)
  unique_labels = unique_labels[counts >=3]
  counts = counts[counts>=3]

  outcome = -1 

  if len(unique_labels) == 3:
    labels, ct = unique_labels[1:], counts[1:]
    max_labels = labels[ np.argwhere(ct == np.amax(ct))][:,0]
    if np.std(ct) >1:
      outcome = max_labels.item()
    else:
      outcome = -1

  elif len(unique_labels) == 2:
    if 1 in unique_labels:
      outcome = max(unique_labels)
    else: 
      if np.std(counts) >1:
        max_labels = unique_labels[ np.argwhere(counts == np.amax(counts))][:,0]
        outcome = max(max_labels).item()
      else:
        outcome = -1
  elif len(unique_labels) == 1:
      outcome = unique_labels.item()
  else:
    outcome =-1
  
  return outcome 

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
  preds = filter_targets_indices(preds, keep, detach=False)
  if len(preds['scores']) >= 1:
    pred_indices = custom_x_nms(preds, iou_thresh=0.4)
    preds = filter_targets_indices(preds, pred_indices, detach=False)
  if len(preds['scores']) >= 1:
    preds = fill_empty_patches(preds)
  if len(preds['scores']) >= 1:
    preds = replace_sandwiched_labels(preds)
  return preds


def get_outcome_from_list(list_name):
  if len(list_name) > 0:
    unique_outcomes, counts = np.unique(list_name, return_counts=True)

    idx = np.argmax(counts)

    if len(unique_outcomes) == 3:
      outcome = unique_outcomes[idx]
      if outcome == 0:
        if counts[1] > 2:
          outcome = 1
        elif counts[2] > 2:
          outcome = 2

    elif len(unique_outcomes) == 2:
      if 0 in unique_outcomes:
        if counts[1] > 2: outcome = max(list_name)
        else: outcome = 0

      else: 
        outcome =unique_outcomes[idx]
    elif len(unique_outcomes) == 1:
        outcome = list_name[0]
    else:
      outcome =-1  
  

  else: return -1 
  
  return outcome

def get_outcome_per_section(x, labels, portion_side, num_sections=4):
  max_l = portion_side
  max_middle = (num_sections - 1) * portion_side
  
  l_left, l_middle, l_right = [], [], []

  x = x.to_list()
  labels = labels.to_list()

  for i in range(len(x)):
    if x[i] < max_l:
      l_left.append(labels[i])
    elif max_l <= x[i] < max_middle:
      l_middle.append(labels[i])
    else:
      l_right.append(labels[i])

  outcome_left = get_outcome_from_list(l_left)
  outcome_middle = get_outcome_from_list(l_middle)
  outcome_right = get_outcome_from_list(l_right)

  return outcome_left, outcome_middle, outcome_right


def compute_eye_bbx(seg, label=1, pad=0.01):

  shape = seg.shape
  
  ij = torch.argwhere(seg.squeeze() != 0)

  bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

  bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
  bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
  bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
  bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
  
  return bb
