from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import os
import glob
import sys
import pandas as pd
import itk
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DatasetGenerator:
    def __init__(self, df):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32), 
            output_shapes=((256, 256, 3), [4])
            )

        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(48)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = row["patch_im"]
            sev = row["sev"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))
            img_np_orig = (np.array(img_np.shape) - np.array((256,256,3)))/np.array([2, 2, 1])
            img_np_orig = img_np_orig.astype(int)
            img_np_end = img_np_orig + np.array([256,256,3])
            img_np_end = img_np_end.astype(int)

            img_np = img_np[img_np_orig[0]:img_np_end[0], img_np_orig[1]:img_np_end[1], img_np_orig[2]:img_np_end[2]]

            yield img_np, tf.one_hot(sev, 4)


gpus_index = [1]
print("Using gpus:", gpus_index)
gpus = tf.config.list_physical_devices('GPU')


if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        gpu_visible_devices = []
        for i in gpus_index:
            gpu_visible_devices.append(gpus[i])
        
        print(bcolors.OKGREEN, "Using gpus:", gpu_visible_devices, bcolors.ENDC)

        tf.config.set_visible_devices(gpu_visible_devices, 'GPU')
        # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(bcolors.FAIL, e, bcolors.ENDC)
# else:
#     # strategy = tf.distribute.get_strategy() 


fname = "/work/jprieto/data/remote/EGower/hinashah/patch_training_07162021/test_patches.csv"
df = pd.read_csv(fname)
checkpoint_path = "/work/jprieto/data/remote/EGower/train/patch_training_08252021"

dataset = DatasetGenerator(df).get()

model = tf.keras.models.load_model(checkpoint_path, custom_objects={'tf': tf})
model.summary()


predictions = model.predict(dataset)

predictions = np.argmax(predictions, axis=1)
df["prediction"] = predictions
df.to_csv(fname.replace(".csv", "_prediction.csv"), index=False)
