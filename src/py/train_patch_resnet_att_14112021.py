from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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
        return (None, 2048)


    def call(self, x):        

        return self.model_features(x)

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        self.features = Features()

        self.V = layers.Dense(256)
        self.A = Attention(128, 1)
        self.P = layers.Dense(3, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.features(x)
        
        v = self.V(x)
        x, w_a = self.A(x, v)
        x = self.P(x)

        return x

class DatasetGenerator:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((448, 448, 3), [3], [1])
            )

        self.dataset = self.dataset.batch(32)
        self.dataset = self.dataset.prefetch(48)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
            patch_class = row["patch_class"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))

            img_np = ndimage.rotate(img_np, np.random.random() * 180 - 90, reshape=False, mode='reflect')
            img_np = img_np[16:-16,16:-16,:]
            img_np = tf.image.random_crop(img_np, size=(448, 448, 3))

            yield img_np, tf.one_hot(patch_class, 3), np.array([self.unique_class_weights[patch_class]])


gpus_index = [3]
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



# df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/test/Patches_02152021/train_patches.csv")
# df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/patch_training_07162021/train_patches.csv")
df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_patches_train.csv")

df.drop(df[df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
df = df.reset_index()
df = df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})

train_df, valid_df = train_test_split(df, test_size=0.1)

unique_classes = np.unique(train_df['patch_class'])
unique_class_weights = np.array(class_weight.compute_class_weight('balanced', unique_classes, train_df['patch_class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", train_df.size, "Valid size:", valid_df.size)

dataset = DatasetGenerator(train_df, unique_class_weights).get()
dataset_validation = DatasetGenerator(valid_df, unique_class_weights).get()

model = TTModel()
# model.summary()

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
# checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3, checkpoint_name=modelname)

# checkpoint_path = "/work/jprieto/data/remote/EGower/train/Patches_02152021"
# checkpoint_path = "/work/jprieto/data/remote/EGower/train/Patches_02152021_endtoend"
checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/patch_training_resnet_att_14112021"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
