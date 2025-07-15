import torch
import numpy as np
import cv2 
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import itertools
from utils import sort_values,filter_targets_indices

colormap = {
  'Reject':(0,0,0),
  'Healthy':(0, 104, 0),
  'Entropion':(0, 0, 153),
  'Overcorrection':(0, 150, 150),
  'Overlapping_area':(128,128, 128),
}

# colormap = {
#   'Reject':(0,0,0),
#   'Healthy':(0, 104, 0),
#   'ECA':(0, 0, 153),
#   'TT':(0, 150, 150),
#   'Overlapping_area':(128,128, 128),
# }

legend_dict = colormap

def plot_boxes(img, targets):
  ax = plt.gca()
  ax.imshow(img)

  boxes = targets['boxes']
  labels = targets['labels']

  for j in range(labels.shape[0]):
    box = boxes[j]
    label = labels[j]

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1

    rect = Rectangle((x1, y1), width, height, fill=False, color='b', linewidth=1.5)
    ax.add_patch(rect)


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


def select_eyelid_seg(seg):
  eyelid_seg = seg.numpy().copy()
  eyelid_seg[eyelid_seg !=3]=0
  eyelid_seg[eyelid_seg ==3]=1
  eyelid_seg = (255*eyelid_seg).astype('uint8')
  return eyelid_seg

def create_mask(eyelid_seg, targets):

  H,W = eyelid_seg.shape
  labels = targets['labels']
  boxes = targets['boxes']

  label_mask = np.zeros((H,W,3), dtype=np.uint8)

  # Create separate masks for each label
  mask_label1 = np.zeros((H,W), dtype=np.uint8)
  mask_label2 = np.zeros((H,W), dtype=np.uint8)
  mask_label3 = np.zeros((H,W), dtype=np.uint8)


  for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = map(int, box)
    if label == 1:
      mask_label1[y1:y2, x1:x2] = 1
    elif label == 2:
      mask_label2[y1:y2, x1:x2] = 1
    elif label == 3:
      mask_label3[y1:y2, x1:x2] = 1

  # Areas with only label 1 (no overlap with 2 or 3)
  only_label1 = mask_label1 & ~mask_label2 & ~mask_label3
  label_mask[only_label1 > 0] = colormap[list(colormap.keys())[1]]# colormap['Healthy']

  # Areas with only label 2 (no overlap with 1 or 3)
  only_label2 = mask_label2 & ~mask_label1 & ~mask_label3
  label_mask[only_label2 > 0] = colormap[list(colormap.keys())[2]]# colormap['Entropion']

  # Areas with only label 3 (no overlap with 1 or 2)
  only_label3 = mask_label3 & ~mask_label1 & ~mask_label2
  label_mask[only_label3 > 0] = colormap[list(colormap.keys())[3]]# colormap['Overcorrection']

  overlap_mask = (mask_label1.astype(int) + mask_label2.astype(int) + mask_label3.astype(int)) > 1
  label_mask[overlap_mask] = colormap[list(colormap.keys())[4]]# colormap['Overlapping_area']

  mask_eyelid = cv2.bitwise_and(label_mask, label_mask, mask=eyelid_seg)

  return label_mask, mask_eyelid

def add_mask_countours(img, overlay):

  img_ui = (255*img).numpy().astype('uint8')
  result = cv2.addWeighted(img_ui, 1.0, overlay, .4, 0)

  out = overlay.copy()
  for key in colormap.keys():

    arr = np.all(out == np.array(colormap[key]), axis=-1).astype(np.uint8) * 255

    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.drawContours(result, contours, -1, colormap[key], 4)

  return result

def create_legend():
  legend = np.ones((200, 300, 3), dtype=np.uint8) * 255
      
  y_pos = 30
  for label, color in colormap:
    cv2.rectangle(legend, (20, y_pos-15), (50, y_pos+15), color, -1)
    cv2.rectangle(legend, (20, y_pos-15), (50, y_pos+15), (0, 0, 0), 1)
    cv2.putText(legend, label, (60, y_pos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    y_pos += 40
  return legend

def fill_empty_patches(targets):

  targets = sort_values(targets)
  boxes = targets['boxes']
  labels = targets['labels']

  avg_size_box = np.average(boxes[:,2] - boxes[:,0])

  # new function : def fill_gap():
  for i in range(boxes.shape[0] -1):

    dist = (boxes[i+1,0] -  boxes[i,2])
    if  dist > avg_size_box:
      num_boxes = int(np.round(dist / avg_size_box).numpy())

      if labels[i] == labels[i+1]:
        y1_step = (boxes[i+1,1] - boxes[i, 1]) / (num_boxes+1)
        y2_step = (boxes[i+1,3] - boxes[i, 3]) / (num_boxes+1)

        for j in range(num_boxes):
          new_coords = torch.tensor([boxes[i,2] + j*avg_size_box, 
                                    boxes[i,1] + (j+2)*y1_step, 
                                    boxes[i,2] + (j+1)*avg_size_box, 
                                    boxes[i,3] + (j+2)*y2_step,  
                                    ])
          targets['boxes'] = torch.cat([targets['boxes'][:i+1+j,:], new_coords.unsqueeze(0), targets['boxes'][i+1+j:,:]])
      
          new_labels = labels[i]
          targets['labels'] = torch.cat([targets['labels'][:i+1+j], new_labels.unsqueeze(0), targets['labels'][i+1+j:]])

          if 'scores' in targets:
            targets['scores'] = targets['scores']
            new_scores = targets['scores'][i:i+2].mean()
            targets['scores'] = torch.cat([targets['scores'][:i+1+j], new_scores.unsqueeze(0), targets['scores'][i+1+j:]])
      # else:
        # idk 
        # print('conflict to raise')

  
  return sort_values(targets)

def replace_sandwiched_labels(targets, context_width=2):    
  targets = sort_values(targets)
  for k in targets.keys():
    targets[k] = targets[k].numpy()
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
        
        else: # create 2 box of half width with each labels
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
          
          if 'scores' in targets:
            scores = targets['scores']
            targets['scores'] = np.insert(scores, i+delta_idx, scores[i+delta_idx], axis=0)

          delta_idx +=1

  targets['boxes'] = updated_boxes
  targets['labels'] = updated_labels
  return targets

def filter_indices_on_segmentation_mask(eyelid_seg, targets, overlap_threshold=0.5):
    out_targets = targets.copy()
    mask = (eyelid_seg > 0).astype(np.uint8)
    filtered_idx = []

    for idx in range(out_targets['boxes'].shape[0]):
        # Extract box coordinates
        x1, y1, x2, y2 = out_targets['boxes'][idx]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        box_mask = mask[y1:y2, x1:x2]
        intersection = np.sum(box_mask)
        box_area = (x2 - x1) * (y2 - y1)
        
        iou = intersection / box_area if box_area > 0 else 0
        
        if iou >= overlap_threshold:
            filtered_idx.append(idx)
    if len(filtered_idx) > 8:
      return filter_targets_indices(out_targets, filtered_idx)
    else:
      return out_targets