
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

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils import shuffle
import pickle  
import SimpleITK as sitk

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
          training = tf.keras.backend.learning_phase()
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
        return tf.image.random_contrast(x, .2, 1.0)
    def hue(self, x):
        return tf.image.random_hue(x, 0.25)
    def brightness(self, x):
        return tf.image.random_brightness(x, 1.)

class RandomAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super(RandomAugmentation, self).__init__()

        self.random_intensity = RandomIntensity()
        self.random_rotation = tf.keras.layers.RandomRotation(0.5)
        self.random_zoom = tf.keras.layers.RandomZoom((-0.3, 1.0))
        # self.resize = tf.keras.layers.Resizing(480, 480)
        self.center_crop0 = tf.keras.layers.CenterCrop(640, 640)
        self.random_crop = tf.keras.layers.RandomCrop(448, 448)        

    def call(self, x):

        x = self.random_intensity(x)
        x = self.random_rotation(x)
        x = self.random_zoom(x)
        x = self.center_crop0(x)
        x = self.random_crop(x)

        return x


class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()

        self.efficient = tf.keras.applications.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        
        self.center_crop = tf.keras.layers.CenterCrop(448, 448)        
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()
        self.augment = RandomAugmentation()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x, training=True):

        if training is None:
            training = tf.keras.backend.learning_phase()
        
        if training:
            x = self.augment(x)
        x = self.center_crop(x)        
        x = self.efficient(x)
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

class TTModel(tf.keras.Model):
    def __init__(self, features = None):
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
        x_a, w_a = self.A(x, x_v)
        
        x = self.P(x_a)

        return x
    

class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, unique_class_weights):
        self.df = df
        self.unique_class_weights = unique_class_weights
    def __len__(self):
        return len(self.df)    
    def __getitem__(self, idx):

        row = self.df.loc[idx]            
        img = os.path.join("/work/jprieto/data/remote/EGower/hinashah/", row["image"])
        sev = row["class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

        one_hot = np.zeros(2)
        one_hot[sev] = 1

        return img_np, one_hot, np.array([self.unique_class_weights[sev]])
        
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/train_stack_efficientv2s_05052022/train_stack_efficientv2s_05052022" 

csv_path_stacks_train = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_20220326/trachoma_bsl_mtss_besrat_field_train_20220326_stacks_train.csv"
train_df = pd.read_csv(csv_path_stacks_train)
train_df['class'] = (train_df['class'] >= 1).astype(int)
train_df = shuffle(train_df)

csv_path_stacks_valid = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_20220326/trachoma_bsl_mtss_besrat_field_train_20220326_stacks_eval.csv"
valid_df = pd.read_csv(csv_path_stacks_valid)
valid_df['class'] = (valid_df['class'] >= 1).astype(int)


unique_classes = np.unique(train_df['class'])
unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_df['class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", len(train_df), "Valid size:", len(valid_df))

dg_train = DatasetGenerator(train_df, unique_class_weights)

def train_generator():
    dg_train.on_epoch_end()
    enqueuer = OrderedEnqueuer(dg_train, shuffle=True, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=256)
    data = enqueuer.get()
    for idx in range(len(dg_train)):
        yield next(data)
    enqueuer.stop()

dataset = tf.data.Dataset.from_generator(train_generator,
    output_signature=(tf.TensorSpec(shape = (None, 768, 768, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (2,), dtype = tf.int32),
        tf.TensorSpec(shape = (1,), dtype = tf.float32))
    )
batch_size = 8
dataset = dataset.batch(batch_size)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
dataset = dataset.prefetch(8)

dg_valid = DatasetGenerator(valid_df, unique_class_weights)

def valid_generator():
    enqueuer = OrderedEnqueuer(dg_valid, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=256)
    data = enqueuer.get()
    for idx in range(len(dg_valid)):
        yield next(data)
    enqueuer.stop()

batch_size_val = 8
dataset_validation = tf.data.Dataset.from_generator(valid_generator,
    output_signature=(tf.TensorSpec(shape = (None, 768, 768, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (2,), dtype = tf.int32),
        tf.TensorSpec(shape = (1,), dtype = tf.float32))
    )
dataset_validation = dataset_validation.batch(batch_size_val)
options_validation = tf.data.Options()
options_validation.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options_validation.threading.private_threadpool_size = 8
dataset_validation = dataset_validation.with_options(options_validation)
dataset_validation = dataset_validation.prefetch(8)


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model_patches = TTModelPatch()    

    model_patches.load_weights("/work/jprieto/data/remote/EGower/jprieto/train/train_patch_efficientv2s_05052022_weights/train_patch_efficientv2s_05052022")

    model = TTModel()
    model.features.efficient = model_patches.features.efficient
    model.features.conv = model_patches.features.conv
    model.features.efficient.trainable = False
    model.features.conv.trainable = False    
    # model.build(input_shape=(None, None, 768, 768, 3))
    # model.summary()    
    
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


