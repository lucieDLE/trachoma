from torchvision.ops import nms
import numpy as np
import torch


colormap = {
  'Healthy':(0, 104, 104),
  'Entropion':(0, 0, 153),
  'Overcorrection':(0, 150, 150),
}


def select_eyelid_seg(seg):
  eyelid_seg = seg.numpy().copy()
  eyelid_seg[eyelid_seg !=3]=0
  eyelid_seg[eyelid_seg ==3]=1
  eyelid_seg = (255*eyelid_seg).astype('uint8')
  return eyelid_seg

def filter_indices_on_segmentation_mask(eyelid_seg, targets, overlap_threshold=0.5):
    out_targets = targets.copy()
    mask = (eyelid_seg > 0).astype(np.uint8)
    keep = []

    for idx in range(out_targets['boxes'].shape[0]):
        # Extract box coordinates
        x1, y1, x2, y2 = out_targets['boxes'][idx]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        box_mask = mask[y1:y2, x1:x2]
        intersection = np.sum(box_mask)
        box_area = (x2 - x1) * (y2 - y1)
        
        iou = intersection / box_area if box_area > 0 else 0
        
        if iou >= overlap_threshold:
            keep.append(idx)
    # if there are too many boxes outside the segmentation mask
    # the segmentation probably have an issue
    if len(keep) > 8:
      for k in out_targets.keys():
        out_targets[k] = out_targets[k][keep]

    return out_targets


def process_predictions(preds):
  for k in preds.keys():
    preds[k] = preds[k].cpu().detach()

  pred_indices = nms(preds['boxes'], preds['scores'], iou_threshold=0.4)
  for k in preds.keys():
    preds[k] = preds[k][pred_indices]

  preds = replace_sandwiched_labels(preds)

  return preds



def replace_sandwiched_labels(targets, context_width=2):
  _, indices = torch.sort(targets['boxes'][:,0])
  for k in targets.keys():
    targets[k] = targets[k][indices].numpy()
  
  boxes = targets['boxes']
  labels = targets['labels']

  updated_labels = labels.copy()
  updated_boxes = boxes.copy()

  delta_idx = 0
  for i in range(len(labels)):
    left_context = labels[max(0, i-context_width):i]
    right_context = labels[i+1:min(len(labels), i+context_width+1)]
    current_label = labels[i]

    # Skip if either context is empty (One box: can't make a decision)
    # or handling for first box and last box of eyelid -> look at following/previous ones
    if len(left_context) == 0 or len(right_context) == 0:
      larger_context = context_width+1
      left_context = labels[max(0, i-larger_context):i]
      right_context = labels[i+1:min(len(labels), i+larger_context+1)]

      if len(left_context) == 0 and len(right_context) >= (larger_context): # 3 boxes needeed
          # First box
          if len(set(right_context)) == 1 and current_label != right_context[0]:
              updated_labels[i+delta_idx] = right_context[0]
      elif len(right_context) == 0 and len(left_context) >= (larger_context):# 3 boxes needeed
          # Last box
          if len(set(left_context)) == 1 and current_label != left_context[0]:
              updated_labels[i+delta_idx] = left_context[0]
      continue

    
    if (current_label not in left_context and current_label not in right_context):
      if (len(set(left_context)) == 1 and  len(set(right_context)) == 1): #only one label

        if (left_context[0] == right_context[0]): #left label == right label            

          # Replace the current label with the context label
          updated_labels[i+delta_idx] = left_context[0]
        
        else: # create 2 boxes of half width with each labels
          x1, y1, x2, y2 = boxes[i]
          xmi = x1 +(x2 -x1)/2

          updated_boxes = np.delete(updated_boxes,i+delta_idx, axis=0)
          updated_labels = np.delete(updated_labels,i+delta_idx)

          box_right = np.array([ xmi, y1, x2, y2], ndmin=2)
          updated_boxes = np.insert(updated_boxes, i+delta_idx, box_right, axis=0)
          updated_labels = np.insert(updated_labels, i+delta_idx, right_context[0], axis=0)

          box_left = np.array([ x1, y1, xmi, y2], ndmin=2)
          updated_boxes = np.insert(updated_boxes, i+delta_idx, box_left, axis=0)
          updated_labels = np.insert(updated_labels, i+delta_idx, left_context[0], axis=0)
          
          scores = targets['scores']
          targets['scores'] = np.insert(scores, i+delta_idx, scores[i+delta_idx], axis=0)

          delta_idx +=1

  targets['boxes'] = updated_boxes
  targets['labels'] = updated_labels
  return targets



def eye_level_outcome(preds,img_shape):
  unique_labels, counts = np.unique(preds['labels'], return_counts=True)
  unique_labels = unique_labels[counts >=3]
  counts = counts[counts>=3] # need at least 3 boxes to be a potential outcome

  outcome = -1 

  if len(unique_labels) == 1: # only one outcome 
      outcome = unique_labels.item()

  elif len(unique_labels) == 2:
    if 1 in unique_labels:
      # if Healthy and one bad outcome -> return the bad outcome
      outcome = max(unique_labels)
    else: 
      # if Under and Over -> either conflict or the most encountered label
      if np.std(counts) >=1:
        max_labels = unique_labels[ np.argwhere(counts == np.amax(counts))][:,0]
        outcome = max(max_labels).item()

  elif len(unique_labels) == 3: 
    # if Healthy, Under, Over -> either conflict or the most encountered label
    labels, ct = unique_labels[1:], counts[1:]
    max_labels = labels[ np.argwhere(ct == np.amax(ct))][:,0]
    if np.std(ct) >=1:
      outcome = max_labels.item()

  return outcome

def plot_patches(img, targets, map, title='ground truth' ):
  ax = plt.gca()
  ax.imshow(img)
  ax.set_title(title)

  boxes = targets['boxes']
  labels = targets['labels']

  for j in range(labels.shape[0]):
    box = boxes[j]
    label = labels[j]

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1

    color = np.array(colormap[map[label.item()]])/255

    rect = Rectangle((x1, y1), width, height, fill=False, color=color, linewidth=1.5)
    ax.add_patch(rect)