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

class Attention(tf.keras.layers.Layer):
    def __init__(self, units, w_units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(w_units)

    def call(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query)))
        
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        x0 = tf.keras.Input(shape=[384, 384, 3])
        self.resnet50 = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=x0, pooling='avg')
        
        self.PreProcess = tf.keras.layers.Rescaling(1/127.5, offset=-1)
        self.TD = layers.TimeDistributed(self.resnet50)

        x0 = tf.keras.Input(shape=[None, 2048])
        
        self.V = layers.Dense(256)

        self.A = Attention(128, 256)
        self.P = layers.Dense(1, activation='sigmoid', name='predictions')
        
        v = self.V(x0)
        x, w_a = self.A(x0, v)
        x = self.P(x)

        self.model_prediction = tf.keras.Model(inputs=x0, outputs=x)

    def call(self, x):

        x = self.PreProcess(x)
        x = self.TD(x)

        x = self.model_prediction(x)

        return x

class DatasetGenerator:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((None, 384, 384, 3), [1], [1])
            )

        self.dataset = self.dataset.batch(8)
        self.dataset = self.dataset.prefetch(16)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = row["img"]
            sev = row["class"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))
            
            img_np = img_np[np.random.choice(64, 8, replace=False)]

            img_np = np.array([ndimage.rotate(im, np.random.random() * 180 - 90, reshape=False, mode='reflect') for im in img_np])
            img_np = img_np[:,16:-16,16:-16,:]
            img_np = tf.image.random_crop(img_np, size=(8, 384, 384, 3))

            yield img_np, np.array([sev]), np.array([self.unique_class_weights[sev]])

class DatasetGeneratorValid:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((None, 384, 384, 3), [1], [1])
            )

        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(16)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = row["img"]
            sev = row["class"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))
            # img_np_orig = (np.array(img_np.shape) - np.array((256,256,3)))/np.array([2, 2, 1])
            # img_np_orig = img_np_orig.astype(int)
            # img_np_end = img_np_orig + np.array([256,256,3])
            # img_np_end = img_np_end.astype(int)

            # img_np = img_np[img_np_orig[0]:img_np_end[0], img_np_orig[1]:img_np_end[1], img_np_orig[2]:img_np_end[2]]
            # img_np = tf.image.random_crop(img_np, size=(256, 256, 3))
            # img_np = img_np[np.random.choice(32, 8, replace=False)]
            
            yield img_np, np.array([sev]), np.array([self.unique_class_weights[sev]])


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

df = pd.read_csv("/work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_64_448_09072021_train.csv")

# group_by = 'im'
# unique_group_valid = df[group_by].unique()
# np.random.shuffle(unique_group_valid)

# samples_validation = round(len(unique_group_valid)*0.1)

# unique_group_valid = unique_group_valid[0:samples_validation]

# train_df = df[~df[group_by].isin(unique_group_valid)]
# valid_df = df[df[group_by].isin(unique_group_valid)]

df['class'] = (df['class'] >= 1).astype(int)
train_df, valid_df = train_test_split(df, test_size=0.1)

unique_classes = np.unique(train_df['class'])
unique_class_weights = np.array(class_weight.compute_class_weight('balanced', unique_classes, train_df['class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", train_df.size, "Valid size:", valid_df.size)

dataset = DatasetGenerator(train_df, unique_class_weights).get()
dataset_validation = DatasetGenerator(valid_df, unique_class_weights).get()

model = TTModel()
model.build(input_shape=[None, None, 384, 384, 3])
model.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"])

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
# checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_10042021_64_448_2class/"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
