from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend
from keras.utils import control_flow_util
import json
import os
import glob
import sys
import pandas as pd
import SimpleITK as sitk
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
        self.random_zoom = tf.keras.layers.RandomZoom((-0.5, 0))
        self.random_crop = tf.keras.layers.RandomCrop(448, 448)
        self.resize = tf.keras.layers.Resizing(512, 512)
        self.center_crop = tf.keras.layers.CenterCrop(448, 448)
        self.rescale = layers.Rescaling(1/127.5, offset=-1)      
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x, training=True):

        if training is None:
          training = backend.learning_phase()

        x = self.random_intensity(x)
        x = self.random_rotation(x)
        x = self.random_zoom(x)
        x = tf.cond(tf.cast(training, tf.bool), 
            lambda: self.resize(x),
            lambda: x)
        x = self.random_crop(x)
        x = self.center_crop(x)
        x = self.rescale(x)
        x = self.resnet(x)
        x = self.conv(x)
        x = self.avg(x)

        return x

class TTModelPatch(tf.keras.Model):
    def __init__(self):
        super(TTModelPatch, self).__init__()

        self.features = Features()
        self.P = layers.Dense(3, activation='softmax', name='predictions')
        
    def call(self, x):

        x_f = self.features(x)
        x = self.P(x_f)

        return x


class TTModel(tf.keras.Model):
    def __init__(self, features = None):
        super(TTModel, self).__init__()

        self.features = Features()
        if(features):
            self.features.resnet = features.resnet
            # self.features.conv = features.conv

        self.TD = layers.TimeDistributed(self.features)
        self.R = layers.Reshape((-1, 512))

        self.V = layers.Dense(256)
        self.A = Attention(128, 1)        
        self.P = layers.Dense(2, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.TD(x)
        x = self.R(x)

        x_v = self.V(x)
        x_a, w_a = self.A(x, x_v)
        
        x = self.P(x_a)

        return x
    

class DatasetGenerator:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((None, 768, 768, 3), [2], [1])
            )

        self.dataset = self.dataset.batch(32)
        self.dataset = self.dataset.prefetch(64)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        self.dataset = self.dataset.with_options(options)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["img"])
            sev = row["class"]

            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

            yield img_np, tf.one_hot(sev, 2), np.array([self.unique_class_weights[sev]])

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_resnet_att_12012022_weights/stack_training_resnet_att_12012022"

csv_path_stacks_train = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_stacks_16_768_train_train.csv"
train_df = pd.read_csv(csv_path_stacks_train).replace("/work/jprieto/data/remote/EGower/", "", regex=True)
train_df['class'] = (train_df['class'] >= 1).astype(int)
train_df = shuffle(train_df)

csv_path_stacks_valid = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_stacks_16_768_train_valid.csv"
valid_df = pd.read_csv(csv_path_stacks_valid).replace("/work/jprieto/data/remote/EGower/", "", regex=True)
valid_df['class'] = (valid_df['class'] >= 1).astype(int)


unique_classes = np.unique(train_df['class'])
unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_df['class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", train_df.size, "Valid size:", valid_df.size)

dataset = DatasetGenerator(train_df, unique_class_weights).get()
dataset_validation = DatasetGenerator(valid_df, unique_class_weights).get()


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model_patches = TTModelPatch()
    model_patches.load_weights("/work/jprieto/data/remote/EGower/jprieto/train/patch_training_resnet_06012022_weights/patch_training_resnet_06012022")

    model = TTModel()
    model.features.resnet = model_patches.features.resnet
    model.features.resnet.trainable = False    
    model.build(input_shape=(None, None, 768, 768, 3))
    model.summary()    
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["acc"])



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True, 
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])

