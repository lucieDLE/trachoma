
import os
import sys
import time
import math
import pandas as pd
import numpy as np
import json

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
        return tf.image.random_saturation(x, 0, 2)
    def contrast(self, x):
        return tf.image.random_contrast(x, .3, 1.0)
    def hue(self, x):
        return tf.image.random_hue(x, 0.25)
    def brightness(self, x):
        return tf.image.random_brightness(x, 1.)

class ColorJitter(tf.keras.layers.Layer):
    def call(self, x):
        x = self.saturation(x)
        x = self.contrast(x)
        x = self.hue(x)
        x = self.brightness(x)
        return x
    def saturation(self, x):
        return tf.image.random_saturation(x, 0, 2)
    def contrast(self, x):
        return tf.image.random_contrast(x, .3, 1.0)
    def hue(self, x):
        return tf.image.random_hue(x, 0.25)
    def brightness(self, x):
        return tf.clip_by_value(tf.image.random_brightness(x, .3), 0.0, 1.0)

class RandomApply(tf.keras.layers.Layer):
    def __init__(self, layer, prob=0.5):
        super(RandomApply, self).__init__()
        self.layer = layer
        self.prob = prob
    def call(self, x):
        p = tf.random.uniform(shape=[], maxval=1.0)
        x = tf.cond(p < self.prob, lambda: self.layer(x), lambda: x)
        return x

class RandomChoose(tf.keras.layers.Layer):
    def __init__(self, layer0, layer1, prob=0.5):
        super(RandomChoose, self).__init__()
        self.layer0 = layer0
        self.layer1 = layer1
        self.prob = prob
    def call(self, x):
        p = tf.random.uniform(shape=[], maxval=1.0)
        x = tf.cond(p < self.prob, lambda: self.layer0(x), lambda: self.layer1(x))
        return x

class RandomMinMaxCrop(tf.keras.layers.Layer):
    def __init__(self, minval, maxval):
        super(RandomMinMaxCrop, self).__init__()
        self.minval = minval
        self.maxval = maxval
    def call(self, x):
        s = tf.random.uniform(shape=[], minval=self.minval, maxval=self.maxval, dtype=tf.int32)
        x = tf.image.random_crop(x, size=(tf.shape(x)[0], s, s, tf.shape(x)[-1]))
        return x

class RandomResizedCrop(tf.keras.layers.Layer):
    def __init__(self):
        super(RandomResizedCrop, self).__init__()        
        self.random_crop = RandomMinMaxCrop(576, 768)
        self.resize = tf.keras.layers.Resizing(448, 448)
    def call(self, x):
        x = self.random_crop(x)        
        x = self.resize(x)
        return x

class RandomCenterCrop(tf.keras.layers.Layer):
    def __init__(self):
        super(RandomCenterCrop, self).__init__()        
        self.center_crop = tf.keras.layers.CenterCrop(512, 512)
        self.random_crop = tf.keras.layers.RandomCrop(448, 448)
        self.random_zoom = tf.keras.layers.RandomZoom((-0.1, 0.0))
    def call(self, x):
        x = self.center_crop(x)        
        x = self.random_crop(x)
        x = self.random_zoom(x)
        return x

class RandomAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super(RandomAugmentation, self).__init__()

        self.gaussian = tf.keras.layers.GaussianNoise(0.01)
        self.color_jitter = ColorJitter()
        self.random_resize = RandomChoose(RandomResizedCrop(), RandomCenterCrop())
        self.random_flip = tf.keras.layers.RandomFlip(mode='horizontal')
        self.random_rotation = tf.keras.layers.RandomRotation(0.5)                

    def call(self, x):

        x = self.gaussian(x)
        x = self.color_jitter(x)
        x = self.random_resize(x)
        x = self.random_flip(x)
        x = self.random_rotation(x)        

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

        self.rescale = layers.Rescaling(1./255)
        self.features = Features()
        self.P = layers.Dense(3, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.rescale(x)
        x_f = self.features(x)
        x = self.P(x_f)

        return x


# class DotProductAttention(tf.keras.layers.Layer):
#     def __init__(self):
#         super(DotProductAttention, self).__init__()
#     def call(self, query, value):
#         input_size = value.shape[1]
#         score = tf.matmul(query, value, transpose_b=True)        
#         attn = tf.nn.softmax(tf.reshape(score, (-1, input_size)), axis=1)
#         attn = tf.reshape(attn, (-1, input_size, input_size))
#         context = tf.matmul(attn, value)        
#         context = tf.reduce_sum(context, axis=1)
#         return context, attn

# class DotProductAttention(tf.keras.layers.Layer):
#     def __init__(self):
#         super(DotProductAttention, self).__init__()
#     def call(self, query, value):
#         input_size = value.shape[1]
#         scores = tf.matmul(query, value, transpose_b=True)        
#         distribution = tf.nn.softmax(scores)
#         context = tf.matmul(attn, value)
#         context = tf.reduce_sum(context, axis=1)
#         return context, attn


# class Attention(tf.keras.layers.Layer):
#     def __init__(self, units, w_units):
#         super(Attention, self).__init__()
#         self.W1 = tf.keras.layers.Dense(units)
#         self.V = tf.keras.layers.Dense(w_units)

#     def call(self, query, values):        

#         # score shape == (batch_size, max_length, 1)
#         # we get 1 at the last axis because we are applying score to self.V
#         # the shape of the tensor before applying self.V is (batch_size, max_length, units)
#         score = self.V(tf.nn.tanh(self.W1(query)))
        
#         attention_weights = tf.nn.softmax(score, axis=1)

#         context_vector = attention_weights * values
#         context_vector = tf.reduce_sum(context_vector, axis=1)

#         return context_vector, attention_weights

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units, w_units):
        super(SelfAttention, self).__init__()
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

        self.rescale = layers.Rescaling(1./255)
        self.features = Features()        

        self.TD = layers.TimeDistributed(self.features)
        self.R = layers.Reshape((-1, 512))

        # self.V = tf.keras.Sequential()
        # self.V.add(layers.Dense(256))
        # self.V.add(layers.ReLU())
        # self.V.add(layers.Dense(1536))
        self.V = layers.Dense(256)

        self.A = SelfAttention(128, 1)
        self.P = layers.Dense(2, activation='softmax', name='predictions')
 
        
    def call(self, x):

        x = self.rescale(x)
        x = self.TD(x)
        x = self.R(x)

        x_v = self.V(x)
        x_a, w_a = self.A(x, x_v)       
        
        x = self.P(x_a)

        return x
    

class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, unique_class_weights, training=False):
        self.df = df
        self.unique_class_weights = unique_class_weights
    def __len__(self):
        return len(self.df)    
    def __getitem__(self, idx):

        row = self.df.loc[idx]            
        img = os.path.join(mount_point, row["image"])
        sev = row["class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

        one_hot = np.zeros(2)
        one_hot[sev] = 1

        return img_np, one_hot, np.array([self.unique_class_weights[sev]])
        
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)


temp_mount_point = "/work/jprieto/data/EGower/jprieto/"
checkpoint_path = os.path.join(temp_mount_point, "train/train_stack_efficientv2s_Analysis_Set_202208/21102022_weights_512")

mount_point = "./"
# train_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_train.csv')
# valid_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_eval.csv')
# # test_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_eval.csv')

# train_df = pd.read_csv(train_fn)
# train_df['class'] = (train_df['class'] >= 1).astype(int)
# train_df = shuffle(train_df)

# valid_df = pd.read_csv(valid_fn)
# valid_df['class'] = (valid_df['class'] >= 1).astype(int)


# unique_classes = np.unique(train_df['class'])
# unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_df['class']))

# print("Unique classes:", unique_classes, unique_class_weights)
# print("Train size:", len(train_df), "Valid size:", len(valid_df))

train_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_train.csv')
valid_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_train_202208_stacks_eval.csv')

class_column = "class"
img_column = "image"
df_train = pd.read_csv(train_fn)    

unique_classes = np.sort(np.unique(df_train[class_column]))
unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[class_column]))

class_replace = {}
for cn, cl in enumerate(unique_classes):
    class_replace[cl] = cn
print(unique_classes, unique_class_weights, class_replace)

df_train[class_column] = df_train[class_column].replace(class_replace).astype(int)

df_val = pd.read_csv(valid_fn)        
df_val[class_column] = df_val[class_column].replace(class_replace).astype(int)


dg_train = DatasetGenerator(df_train, unique_class_weights, training=True)

def train_generator():
    dg_train.on_epoch_end()
    enqueuer = OrderedEnqueuer(dg_train, shuffle=True, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=256)
    data = enqueuer.get()
    for idx in range(len(dg_train)):
        yield next(data)
    enqueuer.stop()

dataset = tf.data.Dataset.from_generator(train_generator,
    output_signature=(tf.TensorSpec(shape = (16, 768, 768, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (2,), dtype = tf.int32),
        tf.TensorSpec(shape = (1,), dtype = tf.float32))
    )
batch_size = 4
dataset = dataset.batch(batch_size)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
dataset = dataset.prefetch(8)

dg_valid = DatasetGenerator(df_val, unique_class_weights)

def valid_generator():
    enqueuer = OrderedEnqueuer(dg_valid, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=256)
    data = enqueuer.get()
    for idx in range(len(dg_valid)):
        yield next(data)
    enqueuer.stop()

batch_size_val = 4
dataset_validation = tf.data.Dataset.from_generator(valid_generator,
    output_signature=(tf.TensorSpec(shape = (16, 768, 768, 3), dtype = tf.float32), 
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
    model_patches.build(input_shape=(None, 768, 768, 3))
    # model_patches.load_weights("train/train_patch_efficientv2s_Analysis_Set_202208/21102022_weights_512/21102022_weights_512_random_crop")
    # model_patches = tf.keras.models.load_model(os.path.join(mount_point, "train/train_patch_efficientv2s_Analysis_Set_202208/21102022"))
    # model = TTModel(model_patches.features)

    model = TTModel()
    # model.features.efficient.trainable = False
    # model.features.conv.trainable = False 
    # model.build(input_shape=(None, None, 768, 768, 3))
    # model.summary()    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["acc"])



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])