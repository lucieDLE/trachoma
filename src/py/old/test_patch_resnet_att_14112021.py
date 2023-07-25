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
from scipy import ndimage
import pickle

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
            output_shapes=((448, 448, 3), [3])
            )

        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(48)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
            patch_class = row["patch_class"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))
            xs, ys, _ = img_np.shape
            xo = (xs - 448)//2
            yo = (ys - 448)//2
            img_np = img_np[xo:xo + 448, yo:yo + 448,:]

            yield img_np, tf.one_hot(patch_class, 3)


gpus_index = [0]
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



output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"
model_path = "/work/jprieto/data/remote/EGower/jprieto/train/patch_training_resnet_att_14112021"
csv_path = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_patches_test.csv"

df = pd.read_csv(csv_path)

df.drop(df[df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
df = df.reset_index()
df = df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})

test_df = df

dataset_test = DatasetGenerator(test_df).get()

model = tf.keras.models.load_model(model_path)
model.summary()

predictions = model.predict(dataset_test, verbose=1)
test_df["prediction"] = list(predictions)
# print(predictions)
with open(os.path.join(output_dir, 'patch_training_resnet_att_14112021.pickle'), 'wb') as f:
    pickle.dump(test_df, f)