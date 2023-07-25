from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.losses import Loss
import json
import os
import glob
import sys
import pandas as pd
import nrrd
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

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        
        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):
        
        x = self.rescale(x)
        x = self.resnet(x)
        x = self.conv(x)
        x = self.avg(x)

        return x

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        self.features = Features()

        self.TD = layers.TimeDistributed(self.features)
        self.R = layers.Reshape((-1, 512))

        self.V = layers.Dense(256)
        self.A = Attention(128, 1)        
        self.P = layers.Dense(2, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.TD(x)
        x = self.R(x)

        x_v = self.V(x)
        x_a, w_s = self.A(x, x_v)
        
        x = self.P(x_a)
        x_v_p = self.P(x_v)

        return x, x_a, x_v, w_s, x_v_p

class DatasetGenerator:
    def __init__(self, df):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32), 
            output_shapes=((None, 448, 448, 3), [2])
            )

        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(128)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["img"])
            sev = row["class"]

            # img_np = itk.GetArrayViewFromImage(itk.imread(img))
            img_np, head = nrrd.read(img, index_order='C')

            if len(img_np.shape) == 3:
                img_np = np.expand_dims(img_np, axis=0)

            t, xs, ys, _ = img_np.shape
            xo = (xs - 448)//2
            yo = (ys - 448)//2
            img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]

            yield img_np, tf.one_hot(sev, 2)


csv_path_stacks = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_stacks_16_544_test.csv"

df_stacks = pd.read_csv(csv_path_stacks)
df_stacks['class'] = (df_stacks['class'] >= 1).astype(int)

dataset_stacks = DatasetGenerator(df_stacks).get()

# model = TTModel()

# model.load_weights(checkpoint_path)

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_resnet_att_28122021_weights/"

# model = tf.keras.models.load_model(model_path, custom_objects={'TTModel': TTModel})
model = TTModel()
model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

    
dataset_stacks_predict = model.predict(dataset_stacks, verbose=True)

output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"


with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_epi_stacks_16_544_test_31122021.pickle'), 'wb') as f:
    pickle.dump(dataset_stacks_predict, f)

csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_16_544_10082021_test.csv"

df_stacks = pd.read_csv(csv_path_stacks)
df_stacks['class'] = (df_stacks['class'] >= 1).astype(int)

dataset_stacks = DatasetGenerator(df_stacks).get()

dataset_stacks_predict = model.predict(dataset_stacks, workers=8, verbose=True)
output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_05182021_stack_16_544_10082021_test_31122021.pickle'), 'wb') as f:
    pickle.dump(dataset_stacks_predict, f)
