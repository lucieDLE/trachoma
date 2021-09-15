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

def make_model():

        x0 = tf.keras.Input(shape=[256, 256, 3])
        x = preprocess_input(x0)

        # resnet50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=x0, pooling='avg', classes=2)
        resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x0, pooling='avg')

        x = resnet50(x)

        x = layers.Dense(4, activation='softmax', name='predictions')(x)

        return tf.keras.Model(inputs=x0, outputs=x)

class DatasetGenerator:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((256, 256, 3), [4], [1])
            )

        self.dataset = self.dataset.batch(32)
        self.dataset = self.dataset.prefetch(48)
        self.unique_class_weights = unique_class_weights


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
            # ImageType = itk.VectorImage[itk.F, 3]
            # img_read = itk.ImageFileReader[ImageType].New(FileName=img)
            # img_read.Update()
            # img_np = itk.GetArrayViewFromImage(img_read.GetOutput())
            # img_np = img_np.reshape([12, 512, 512, 4])

            # ImageType = itk.Image[itk.UC, 3]
            # img_read_seg = itk.ImageFileReader[ImageType].New(FileName=seg)
            # img_read_seg.Update()
            # seg_np = itk.GetArrayViewFromImage(img_read_seg.GetOutput())
            # seg_np = seg_np.reshape([12, 512, 512])

            yield img_np, tf.one_hot(sev, 4), np.array([self.unique_class_weights[sev]])


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



# df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/test/Patches_02152021/train_patches.csv")
df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/patch_training_07162021/train_patches.csv")

# group_by = 'im'
# unique_group_valid = df[group_by].unique()
# np.random.shuffle(unique_group_valid)

# samples_validation = round(len(unique_group_valid)*0.1)

# unique_group_valid = unique_group_valid[0:samples_validation]

# train_df = df[~df[group_by].isin(unique_group_valid)]
# valid_df = df[df[group_by].isin(unique_group_valid)]

train_df, valid_df = train_test_split(df, test_size=0.1)

unique_classes = np.unique(train_df['sev'])
unique_class_weights = np.array(class_weight.compute_class_weight('balanced', unique_classes, train_df['sev']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", train_df.size, "Valid size:", valid_df.size)

dataset = DatasetGenerator(train_df, unique_class_weights).get()
dataset_validation = DatasetGenerator(valid_df, unique_class_weights).get()

model = make_model()
model.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
# checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

# checkpoint_path = "/work/jprieto/data/remote/EGower/train/Patches_02152021"
# checkpoint_path = "/work/jprieto/data/remote/EGower/train/Patches_02152021_endtoend"
checkpoint_path = "/work/jprieto/data/remote/EGower/train/patch_training_08252021"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
