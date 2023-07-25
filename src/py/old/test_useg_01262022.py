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


csv_fn = "/work/jprieto/data/remote/EGower/jprieto/eyes_cropped_resampled_512_seg_test.csv"

df = pd.read_csv(csv_fn)


test_df = df

dg_test = DatasetGenerator(test_df)

def test_generator():
    enqueuer = OrderedEnqueuer(dg_test, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=256)
    data = enqueuer.get()
    for idx in range(len(dg_test)):
        yield next(data)
    enqueuer.stop()

dataset_test = tf.data.Dataset.from_generator(test_generator,
    output_signature=(tf.TensorSpec(shape = (512, 512, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (512, 512, 1), dtype = tf.int32))
    )
dataset_test = dataset_test.batch(1)
options_validation = tf.data.Options()
options_validation.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset_test = dataset_test.with_options(options_validation)
dataset_test = dataset_test.prefetch(16)


checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/eyes_cropped_resampled_512_seg_train_random_rot_unet_xception/eyes_cropped_resampled_512_seg_train_random_rot_unet_xception_weights"


model = make_model()
model.summary()

model.load_weights(checkpoint_path)


predictions = []
for idx, (img, seg) in enumerate(dataset_test):

    img_fn = test_df.loc[idx]["img"]
    out_fn = os.path.join("segmentation_predict_tf_xception_model01252022", os.path.basename(img_fn))
    
    predictions.append(out_fn)

    pred = model.predict(img)[0].astype(np.ubyte)
    pred = tf.expand_dims(tf.argmax(pred, axis=-1), axis=-1)

    print("Writing", out_fn)
    sitk.WriteImage(sitk.GetImageFromArray(pred, isVector=True), out_fn)



test_df["prediction"] = predictions
test_df.to_csv(csv_fn.replace(".csv", "_tf_xception_prediction.csv"), index=False)