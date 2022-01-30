import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import OrderedEnqueuer
from tensorflow import keras

import json
import os
import glob
import sys
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
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

def IoU(y_true, y_pred):

    num_classes = 4

    # weights = np.array([0.5, 1, 1, 1, 2])
    y_true = tf.one_hot(y_true, num_classes)
    y_true = tf.cast(y_true, tf.float32)    

    y_true = tf.reshape(y_true, [-1, num_classes])
    y_pred = tf.reshape(y_pred, [-1, num_classes])
    intersection = 2.0*tf.reduce_sum(y_true * y_pred, axis=0) + 1
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) + 1.

    iou = 1.0 - intersection / union

    return tf.reduce_sum(iou)

def make_model(img_size=(512, 512, 3), num_classes=4):
    inputs = tf.keras.Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256, 512, 1024]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [1024, 512, 256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, training=False):
        self.df = df
        self.training = training

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.loc[idx]
            
        img = row["img"]
        seg = row["seg"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))
        seg_np = sitk.GetArrayFromImage(sitk.ReadImage(seg))
        seg_np = seg_np.reshape([512, 512, 1])

        degree = 0
        if(self.training):
            degree = np.random.random() * 180 - 90 
            img_np = ndimage.rotate(img_np, degree, reshape=False, 
                mode='reflect') 
            seg_np = ndimage.rotate(seg_np, degree, reshape=False,
                mode='constant', order=0, cval=0)

        return img_np, seg_np


df = pd.read_csv("/work/jprieto/data/remote/EGower/jprieto/eyes_cropped_resampled_512_seg_train.csv")

train_df, valid_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("Train size:", len(train_df), "Valid size:", len(valid_df))

dg_train = DatasetGenerator(train_df, training=True)

def train_generator():
    dg_train.on_epoch_end()
    enqueuer = OrderedEnqueuer(dg_train, shuffle=True, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=256)
    data = enqueuer.get()
    for idx in range(len(dg_train)):
        yield next(data)
    enqueuer.stop()

dataset = tf.data.Dataset.from_generator(train_generator,
    output_signature=(tf.TensorSpec(shape = (512, 512, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (512, 512, 1), dtype = tf.int32))
    )
dataset = dataset.batch(32)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
dataset = dataset.prefetch(16)

dg_valid = DatasetGenerator(valid_df)

def valid_generator():
    enqueuer = OrderedEnqueuer(dg_valid, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=256)
    data = enqueuer.get()
    for idx in range(len(dg_valid)):
        yield next(data)
    enqueuer.stop()

dataset_validation = tf.data.Dataset.from_generator(valid_generator,
    output_signature=(tf.TensorSpec(shape = (512, 512, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (512, 512, 1), dtype = tf.int32))
    )
dataset_validation = dataset_validation.batch(32)
options_validation = tf.data.Options()
options_validation.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset_validation = dataset_validation.with_options(options_validation)
dataset_validation = dataset_validation.prefetch(16)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = make_model()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(1e-4)
    # model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])
    model.compile(optimizer=optimizer, loss=IoU, metrics=['acc'])


checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/eyes_cropped_resampled_512_seg_train_random_rot_unet_xception/eyes_cropped_resampled_512_seg_train_random_rot_unet_xception_weights"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
