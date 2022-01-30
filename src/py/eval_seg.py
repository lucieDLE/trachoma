
from __future__ import print_function
import numpy as np
import argparse
import os
from datetime import datetime, time
import json
import glob
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp
import pandas as pd
import SimpleITK as sitk
import seaborn as sns

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
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
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

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

  return cm


def main(args):
  df = pd.read_csv(args.csv)

  y_pred_arr = []
  y_true_arr = []
  dice_arr = []

  fpr_arr = []
  tpr_arr = []
  roc_auc_arr = []
  iou_arr = []

  abs_diff_arr = []
  mse_arr = []

  fpr_obj = {}
  tpr_obj = {}
  roc_auc_obj = {}

  for i, row in df.iterrows():

    print("Reading:", row["seg"])
    y_true = sitk.GetArrayFromImage(sitk.ReadImage(row["seg"]))
    print("Reading:", row["prediction"])
    y_pred = sitk.GetArrayFromImage(sitk.ReadImage(row["prediction"]))

    y_pred = np.reshape(y_pred, -1)
    y_true = np.reshape(y_true, -1)

    y_pred_arr.extend(y_pred)
    y_true_arr.extend(y_true)

    jaccard = jaccard_score(y_true, y_pred, average=None)
    dice = 2.0*jaccard/(1.0 + jaccard)
    if(len(dice) == 4):
      print(dice)
      dice_arr.append(dice)
    else:
      print("WTF!")


  cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
  cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
  print(cnf_matrix)
  FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
  FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
  TP = np.diag(cnf_matrix)
  TN = cnf_matrix.sum() - (FP + FN + TP)

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  # Precision or positive predictive value
  PPV = TP/(TP+FP)
  # Negative predictive value
  NPV = TN/(TN+FN)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  # False negative rate
  FNR = FN/(TP+FN)
  # False discovery rate
  FDR = FP/(TP+FP)

  # Overall accuracy
  ACC = (TP+TN)/(TP+FP+FN+TN)

  print("True positive rate or sensitivity:", TPR)
  print("True negative rate or specificity:", TNR)
  print("Positive predictive value or precision:", PPV)
  print("Negative predictive value:", NPV)
  print("False positive rate or fall out", FPR)
  print("False negative rate:", FNR)
  print("False discovery rate:", FDR)
  print("Overall accuracy:", ACC)

  print(classification_report(y_true_arr, y_pred_arr))

  jaccard = jaccard_score(y_true_arr, y_pred_arr, average=None)
  print("jaccard score:", jaccard)
  print("dice:", 2.0*jaccard/(1.0 + jaccard))


  dice_arr = np.array(dice_arr)
  print(dice_arr.shape)
  
  fig3 = plt.figure() 
  # Creating plot
  np.save(os.path.splitext(args.csv)[0] + "_violin_plot.npy", dice_arr)
  s = sns.violinplot(data=dice_arr, cut=0, scale="count")
  # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"])
  s.set_title('Dice coefficients')
  violin_filename = os.path.splitext(args.csv)[0] + "_violin_plot.png"
  fig3.savefig(violin_filename)

  # roc_fig = plt.figure()
  # lw = 3
  # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  # plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.05])
  # plt.xlabel('False Positive Rate')
  # plt.ylabel('True Positive Rate')
  # plt.title('Receiver operating characteristic')


  # for pos_label, fpr_arr, tpr_arr in enumerate(zip(fpr_obj, tpr_obj)):

  #   # First aggregate all false positive rates
  #   all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_arr]))

  #   # Then interpolate all ROC curves at this points
  #   mean_tpr = np.zeros_like(all_fpr)
  #   for i in range(len(fpr_arr)):
  #       mean_tpr += interp(all_fpr, fpr_arr[i], tpr_arr[i])

  #   mean_tpr /= len(fpr_arr)

  #   roc_auc = auc(all_fpr, mean_tpr)
  #   if pos_label < 3:
  #     color = "tab:red"
  #     if pos_label == 2:
  #       color = "tab:green"
  #     elif pos_label == 3:
  #       color = "tab:blue"
  #     plt.plot(all_fpr, mean_tpr, lw=lw, color=color, label='ROC curve of label {0} (area = {1:0.2f})'''.format(pos_label, roc_auc))  
  #   else:
  #     plt.plot(all_fpr, mean_tpr, lw=lw, label='ROC curve of label {0} (area = {1:0.2f})'''.format(pos_label, roc_auc))
      
    
  #   plt.legend(loc="lower right")

  # plt.show()
  # roc_filename = os.path.splitext(args.csv)[0] + "_roc.png"
  # roc_fig.savefig(roc_filename)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Evaluate a model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  input_param_group = parser.add_argument_group('Input')
  input_param_group.add_argument('--csv', type=str, help='csv file columns seg,prediction', required=True)

  args = parser.parse_args()
  main(args)



