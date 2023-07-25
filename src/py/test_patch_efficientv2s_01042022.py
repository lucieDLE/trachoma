
import os
import sys
import time
import math
import pandas as pd
import numpy as np
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import SequenceEnqueuer
from tensorflow.keras.utils import OrderedEnqueuer

import pickle  
import SimpleITK as sitk

from scipy import ndimage

import pickle
from sklearn.metrics import classification_report

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

        return context_vector, score


class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()

        self.efficient = tf.keras.applications.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        
        self.center_crop = tf.keras.layers.CenterCrop(448, 448)        
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):
        
        x = self.center_crop(x)        
        x = self.efficient(x)
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

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)



checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/train_patch_efficient_tf_01042022_weights/train_patch_efficient_tf_01042022"

model = TTModel()
model.load_weights(checkpoint_path)


csv_path = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_20220326/trachoma_bsl_mtss_besrat_field_patches_test_20220326.csv"
test_df = pd.read_csv(csv_path)
test_df.drop(test_df[test_df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
test_df = test_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
test_df = test_df.replace({'/work/hinashah/data/EGower/': ''}, regex=True)
test_df = test_df.reset_index(drop=True)


dg_test = DatasetGenerator(test_df)

def test_generator():

    enqueuer = OrderedEnqueuer(dg_test, use_multiprocessing=True)
    enqueuer.start(workers=8, max_queue_size=128)

    datas = enqueuer.get()

    for idx in range(len(dg_test)):
        yield next(datas)

    enqueuer.stop()

dataset = tf.data.Dataset.from_generator(test_generator,
    output_signature=(tf.TensorSpec(shape = (448, 448, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (3,), dtype = tf.int32))
    )

dataset = dataset.batch(1)
dataset = dataset.prefetch(16)


dataset_predict = model.predict(dataset, verbose=True)


with open(csv_path.replace(".csv", "_01042022_prediction.pickle"), 'wb') as f:
    pickle.dump(dataset_predict, f)

print(classification_report(test_df["patch_class"], np.argmax(dataset_predict[0], axis=1)))


