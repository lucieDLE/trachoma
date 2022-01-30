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

class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()
        x0 = tf.keras.Input(shape=[448, 448, 3])

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[224, 224, 3]), pooling='avg')

        x = tf.keras.layers.Rescaling(1/127.5, offset=-1)(x0)
        q0 = tf.expand_dims(self.resnet(x[:,0:224,0:224,:]), axis=1)
        q1 = tf.expand_dims(self.resnet(x[:,224:,0:224,:]), axis=1)
        q2 = tf.expand_dims(self.resnet(x[:,0:224,224:,:]), axis=1)
        q3 = tf.expand_dims(self.resnet(x[:,224:,224:,:]), axis=1)

        x = layers.Concatenate(axis=1)([q0, q1, q2, q3])

        self.model_features = tf.keras.Model(inputs=x0, outputs=x)
        self.model_features.summary()

    def compute_output_shape(self, input_shape):
        return (None, 1280)


    def call(self, x):        

        return self.model_features(x)

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        self.features = Features()

        self.V = layers.Dense(256, input_shape=[None, 1280])
        self.A = Attention(128, 1)
        self.P = layers.Dense(3, activation='softmax', name='predictions')

    def compute_output_shape(self, input_shape):
        return (None, 3)
        
    def call(self, x):

        x = self.features(x)
        
        v = self.V(x)
        x, w_a = self.A(x, v)
        x = self.P(x)

        return x

class DatasetGenerator:
    def __init__(self, df):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32), 
            output_shapes=((None, 448, 448, 3), [2])
            )

        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(48)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/jprieto/", row["img"])
            seq_class = row["class"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))
            t, xs, ys, _ = img_np.shape
            xo = (xs - 448)//2
            yo = (ys - 448)//2
            img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]

            yield img_np, tf.one_hot(seq_class, 2)


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
csv_path = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_stack_16_544_test.csv"

df = pd.read_csv(csv_path)

df.drop(df[df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
df = df.reset_index()
df = df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})

test_df = df

dataset_test = DatasetGenerator(test_df).get()

model_trained = tf.keras.models.load_model(model_path, custom_objects={'TTModel': TTModel})


tt_model = TTModel()
tt_model.features = model_trained.features
tt_model.V = model_trained.V
tt_model.A = model_trained.A
tt_model.P = model_trained.P

inputs = tf.keras.Input(shape=(None, 448, 448, 3))
outputs = tf.keras.layers.TimeDistributed(tt_model)(inputs)
outputs = tf.math.argmax(outputs, axis=2)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


predictions = model.predict(dataset_test, verbose=1)
test_df["prediction"] = list(predictions)
# print(predictions)
with open(os.path.join(output_dir, 'patch_training_resnet_att_14112021_timedistributed.pickle'), 'wb') as f:
    pickle.dump(test_df, f)