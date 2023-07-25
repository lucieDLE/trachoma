from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
# from keras.utils import control_flow_util

import json
import os
import glob
import sys
import pandas as pd
import itk
# import PIL
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils import shuffle
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

class RandomIntensity(tf.keras.layers.Layer):
    def call(self, x, training=True):
        if training is None:
          training = backend.learning_phase()
        # return control_flow_util.smart_cond(training, 
        return tf.cond(tf.cast(training, tf.bool),
            lambda: self.random_intensity(x),
            lambda: x)
    def random_intensity(self, x):
        r = tf.random.uniform(shape=[], maxval=5, dtype=tf.int32)
        x = tf.cond(r == 1, lambda: self.saturation(x), lambda: x)
        x = tf.cond(r == 2, lambda: self.contrast(x), lambda: x)
        x = tf.cond(r == 3, lambda: self.hue(x), lambda: x)
        x = tf.cond(r == 4, lambda: self.brightness(x), lambda: x)
        return x
    def saturation(self, x):
        return tf.image.random_saturation(x, 0, 10)
    def contrast(self, x):
        return tf.image.random_contrast(x, 0, 0.5)
    def hue(self, x):
        return tf.image.random_hue(x, 0.25)
    def brightness(self, x):
        return tf.image.random_brightness(x, 70)

class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)

        self.random_intensity = RandomIntensity()
        self.random_rotation = tf.keras.layers.RandomRotation(0.5)
        self.random_zoom = tf.keras.layers.RandomZoom(0.1, fill_mode='reflect')
        self.random_crop = tf.keras.layers.RandomCrop(448, 448)
        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.conv0 = layers.Conv2D(1024, (2, 2), strides=(2, 2), padding='same', activation='relu')
        self.conv1 = layers.Conv2D(512, (2, 2), strides=(2, 2), padding='same', activation='relu')
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):

        x = self.random_intensity(x)
        x = self.random_rotation(x)
        x = self.random_zoom(x)
        x = self.random_crop(x)
        x = self.rescale(x)
        x = self.resnet(x)
        x = self.conv0(x)
        x = self.conv1(x)        
        x = self.avg(x)

        return x

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        self.features = Features()
        self.P = layers.Dense(3, activation='softmax', name='predictions')
        
    def call(self, x):

        x_f = self.features(x)
        x = self.P(x_f)

        return x

class DatasetGenerator:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((512, 512, 3), [3], [1])
            )

        self.dataset = self.dataset.batch(128)
        self.dataset = self.dataset.prefetch(32)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        self.dataset = self.dataset.with_options(options)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
            patch_class = row["patch_class"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))

            xs, ys, _ = img_np.shape
            xo = (xs - 512)//2
            yo = (ys - 512)//2
            img_np = img_np[xo:xo + 512, yo:yo + 512,:]

            yield img_np, tf.one_hot(patch_class, 3), np.array([self.unique_class_weights[patch_class]])

class DatasetGeneratorValid:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((448, 448, 3), [3], [1])
            )

        self.dataset = self.dataset.batch(128)
        self.dataset = self.dataset.prefetch(32)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        self.dataset = self.dataset.with_options(options)
        self.unique_class_weights = unique_class_weights


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

            yield img_np, tf.one_hot(patch_class, 3), np.array([self.unique_class_weights[patch_class]])


train_df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_patches_train_train.csv")
train_df.drop(train_df[train_df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
train_df = train_df.reset_index()
train_df = train_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
train_df = shuffle(train_df)

valid_df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_patches_train_valid.csv")
valid_df.drop(valid_df[valid_df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
valid_df = valid_df.reset_index()
valid_df = valid_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})


unique_classes = np.unique(train_df['patch_class'])
unique_class_weights = np.array(class_weight.compute_class_weight('balanced', unique_classes, train_df['patch_class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", train_df.size, "Valid size:", valid_df.size)

dataset = DatasetGenerator(train_df, unique_class_weights).get()
dataset_validation = DatasetGeneratorValid(valid_df, unique_class_weights).get()


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = TTModel()
    model.build(input_shape=(None, 448, 448, 3))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["acc"])

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/patch_training_resnet_10012022_weights/patch_training_resnet_10012022"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
