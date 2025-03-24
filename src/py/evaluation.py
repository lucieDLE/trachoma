import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
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

  
def eye_level_outcome(targets_labels):
  unique_labels, counts = np.unique(targets_labels, return_counts=True)
  unique_labels = unique_labels[counts >3]
  counts = counts[counts >3]

  if len(unique_labels) == 3:
    labels, ct = unique_labels[1:], counts[1:]
    max_labels = labels[ np.argwhere(ct == np.amax(ct))][:,0]
    if np.std(ct) >1:
      return max_labels.item()
    else:
      return -1

  elif len(unique_labels) == 2:
    if 1 in unique_labels:
      return max(unique_labels)
    else: 
      if np.std(counts) >1:
        max_labels = unique_labels[ np.argwhere(counts == np.amax(counts))][:,0]
        return max(max_labels).item()
      else:
        return -1
  else:
      return unique_labels.item()