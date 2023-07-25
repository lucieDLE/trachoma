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

    # weights = np.array([0.5, 1, 1, 1, 2])
    y_true = tf.cast(y_true, tf.float32)
    num_classes = tf.shape(y_true)[-1]

    y_true = tf.reshape(y_true, [-1, num_classes])
    y_pred = tf.reshape(y_pred, [-1, num_classes])
    intersection = 2.0*tf.reduce_sum(y_true * y_pred, axis=0) + 1
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) + 1.

    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])

    # intersection = 2.0*tf.reduce_sum(y_true * y_pred) + 1
    # union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.

    iou = 1.0 - intersection / union
    # iou *= weights

    return tf.reduce_sum(iou)


def make_model():

        drop_prob=0.1

        x0 = tf.keras.Input(shape=[512, 512, 3])

        x = tf.keras.layers.GaussianNoise(1.0)(x0)

        x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d0 = x

        x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d1 = x

        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d2 = x

        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d3 = x

        x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d4 = x

        x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)
        d5 = x

        x = layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d5])
        x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d4])
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d3])
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d2])
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop_prob)(x)

        x = layers.Concatenate(axis=-1)([x, d1])
        x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Concatenate(axis=-1)([x, d0])
        x = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', activation='softmax')(x)

        # if argmax:
        #     x = tf.expand_dims(tf.math.argmax(x, axis=-1), axis=-1)
        #     x = tf.cast(x, dtype=tf.uint8)

        return tf.keras.Model(inputs=x0, outputs=x)

class DatasetGenerator:
    def __init__(self, df):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32), 
            output_shapes=((512, 512, 3), [512, 512, 4])
            )

        self.dataset = self.dataset.batch(16)
        self.dataset = self.dataset.prefetch(16)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = row["img"]
            seg = row["seg"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))
            seg_np = itk.GetArrayViewFromImage(itk.imread(seg))

            degree = np.random.random() * 180 - 90 
            img_np = ndimage.rotate(img_np, degree, reshape=False, 
                mode='reflect') 
            seg_np = ndimage.rotate(seg_np, degree, reshape=False,
                mode='constant', order=0, cval=0)

            seg_np = seg_np.reshape([512, 512])

            yield img_np, tf.one_hot(seg_np, 4)


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


# df = pd.read_csv("/work/jprieto/data/remote/EGower/jprieto/entropion_seg_train.csv")
# df = pd.read_csv("/work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_08162021_train_seg.csv")
df = pd.read_csv("/work/jprieto/data/remote/EGower/jprieto/eyes_cropped_resampled_512_seg_train.csv")

train_df, valid_df = train_test_split(df, test_size=0.1)

print("Train size:", train_df.size, "Valid size:", valid_df.size)

dataset = DatasetGenerator(train_df).get()
dataset_validation = DatasetGenerator(valid_df).get()

model = make_model()
model.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)
# model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

model.compile(optimizer=optimizer, loss=IoU, metrics=['acc'])


checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/eyes_cropped_resampled_512_seg_train_random_rot"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
