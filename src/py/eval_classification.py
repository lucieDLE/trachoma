
from __future__ import print_function
import numpy as np
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import itk
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp

parser = argparse.ArgumentParser(description='Resnet HELPER TEST', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--csv', type=str, help='csv file', required=True)
parser.add_argument('--csv_true_column', type=str, help='Which column to do the stats on', default="class")
parser.add_argument('--csv_prediction_column', type=str, help='csv true class', default='prediction')
parser.add_argument('--class_names', type=str, nargs='+', help='Class names, otherwise guessed from true column', default=None)
parser.add_argument('--out', type=str, help='Output name', default="out.png")
parser.add_argument('--title', type=str, help='Title for the image', default="Confusion matrix")

args = parser.parse_args()

y_true_arr = [] 
y_pred_arr = []

df = pd.read_csv(args.csv)

if args.class_names:
  class_names = args.class_names
else:
  class_names = np.sort(pd.unique(df[args.csv_true_column]))

print(class_names)

for idx, row in df.iterrows():
  y_true_arr.append(row[args.csv_true_column])
  y_pred_arr.append(row[args.csv_prediction_column])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
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
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.xlabel('True label')
  plt.ylabel('Predicted label')
  plt.tight_layout()

  return cm

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix
fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)
confusion_filename = os.path.splitext(args.out)[0] + "_confusion.png"
fig.savefig(confusion_filename)
# Plot normalized confusion matrix
fig2 = plt.figure()
cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')

# cnf_matrix = np.array(cnf_matrix)
# FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
# FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
# TP = np.diag(cnf_matrix)
# TN = cnf_matrix.values.sum() - (FP + FN + TP)

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP) 
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)

# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)

# print("True positive rate:", TPR)
# print("True negative rate:", TNR)
# print("Precision or positive predictive value:", PPV)
# print("Negative predictive value:", NPV)
# print("False positive rate or fall out", FPR)
# print("False negative rate:", FNR)
# print("False discovery rate:", FDR)
# print("Overall accuracy:", ACC)

print(classification_report(y_true_arr, y_pred_arr))

norm_confusion_filename = os.path.splitext(args.out)[0] + "_norm_confusion.json"
with open(norm_confusion_filename, 'w') as outjson:
  json.dump(cm.tolist(), outjson, sort_keys=True, indent=4)

norm_confusion_filename = os.path.splitext(args.out)[0] + "_norm_confusion.png"
fig2.savefig(norm_confusion_filename)



