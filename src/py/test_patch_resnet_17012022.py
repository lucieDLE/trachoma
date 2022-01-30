from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.utils import OrderedEnqueuer

import json
import os
import glob
import sys
import pandas as pd
import SimpleITK as sitk
# import PIL
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils import shuffle
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

# class RandomIntensity(tf.keras.layers.Layer):
#     def call(self, x, training=True):
#         if training is None:
#           training = backend.learning_phase()
#         # return control_flow_util.smart_cond(training, 
#         return tf.cond(tf.cast(training, tf.bool),
#             lambda: self.random_intensity(x),
#             lambda: x)
#     def random_intensity(self, x):
#         r = tf.random.uniform(shape=[], maxval=5, dtype=tf.int32)
#         x = tf.cond(r == 1, lambda: self.saturation(x), lambda: x)
#         x = tf.cond(r == 2, lambda: self.contrast(x), lambda: x)
#         x = tf.cond(r == 3, lambda: self.hue(x), lambda: x)
#         x = tf.cond(r == 4, lambda: self.brightness(x), lambda: x)
#         return x
#     def saturation(self, x):
#         return tf.image.random_saturation(x, 0, 10)
#     def contrast(self, x):
#         return tf.image.random_contrast(x, 0, 0.5)
#     def hue(self, x):
#         return tf.image.random_hue(x, 0.25)
#     def brightness(self, x):
#         return tf.image.random_brightness(x, 70)

class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)

        # self.random_intensity = RandomIntensity()
        # self.random_rotation = tf.keras.layers.RandomRotation(0.5)
        # self.random_zoom = tf.keras.layers.RandomZoom(0.1, fill_mode='reflect')
        # self.random_crop = tf.keras.layers.RandomCrop(448, 448)
        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):

        # x = self.random_intensity(x)
        # x = self.random_rotation(x)
        # x = self.random_zoom(x)
        # x = self.random_crop(x)
        x = self.rescale(x)
        x = self.resnet(x)
        x = self.conv(x)
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

        return x, x_f


class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.loc[idx]
            
        img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
        patch_class = row["patch_class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

        xs, ys, _ = img_np.shape
        xo = (xs - 448)//2
        yo = (ys - 448)//2
        img_np = img_np[xo:xo + 448, yo:yo + 448,:]

        one_hot = np.zeros(3)
        one_hot[patch_class] = 1

        return img_np, one_hot


test_df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_01132022/trachoma_normals_healthy_sev123_epi_patches_test.csv")

test_df.drop(test_df[test_df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
test_df = test_df.reset_index()
test_df = test_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})

dg_test = DatasetGenerator(test_df)


def test_generator():
    enqueuer = OrderedEnqueuer(dg_test, use_multiprocessing=True)
    enqueuer.start(workers=8, max_queue_size=256)
    datas = enqueuer.get()
    for idx in range(len(dg_test)):
        yield next(datas)
    enqueuer.stop()


dataset_test = tf.data.Dataset.from_generator(test_generator,
    output_signature=(tf.TensorSpec(shape = (448, 448, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (3,), dtype = tf.float32))
    )

dataset_test = dataset_test.batch(1)
dataset_test = dataset_test.prefetch(4)


checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/patch_training_resnet_17012022_weights/patch_training_resnet_17012022"

model = TTModel()
model.load_weights(checkpoint_path)

output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

predictions = model.predict(dataset_test, verbose=1)
# test_df["prediction"] = list(predictions)
print(len(test_df), len(predictions))
with open(os.path.join(output_dir, 'patch_training_resnet_17012022.pickle'), 'wb') as f:
    pickle.dump(predictions, f)
