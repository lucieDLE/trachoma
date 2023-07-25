import os
import sys
import time

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
        self.random_crop = RandomMinMaxCrop(512, 768)
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


# class Features(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Features, self).__init__()

#         self.efficient = tf.keras.applications.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        
#         self.center_crop = tf.keras.layers.CenterCrop(448, 448)        
#         self.conv = layers.Conv2D(1536, (2, 2), strides=(2, 2))
#         # self.bn = tf.keras.layers.BatchNormalization()
#         self.avg = layers.GlobalAveragePooling2D()
#         self.augment = RandomAugmentation()

#     def compute_output_shape(self, input_shape):
#         return (None, 1536)


#     def call(self, x, training=True):

#         if training is None:
#             training = tf.keras.backend.learning_phase()
        
#         if training:
#             x = self.augment(x)
#         x = self.center_crop(x)        
#         x = self.efficient(x)
#         x = self.conv(x)
#         # x = self.bn(x)
#         x = self.avg(x)

#         return x

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

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        self.rescale = layers.Rescaling(1./255)
        self.features = Features()
        self.P = layers.Dense(3, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.rescale(x)
        x_f = self.features(x)
        x = self.P(x_f)

        return x

class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, unique_class_weights, training=False, mount_point="./"):
        self.df = df        
        self.unique_class_weights = unique_class_weights
        self.training=training
        self.mount_point = mount_point

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.loc[idx]            
        img = os.path.join(self.mount_point, row["image"])
        patch_class = row["patch_class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

        one_hot = np.zeros(3)
        one_hot[patch_class] = 1

        return img_np, one_hot, np.array([self.unique_class_weights[patch_class]])

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)




mount_point = "./"

print(bcolors.OKBLUE, "Training clean data", bcolors.ENDC)

temp_mount_point = "/work/jprieto/data/EGower/jprieto/"
checkpoint_path = os.path.join(temp_mount_point, "train/train_patch_efficientv2s_Analysis_Set_202208/21102022_weights_512/21102022_weights_512_random_crop")

train_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_patches_train_202208_train.csv')
valid_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_patches_train_202208_eval.csv')
test_fn = os.path.join(mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_patches_test_202208.csv')

train_df = pd.read_csv(train_fn)
train_df.drop(train_df[train_df['patch_class'].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)
train_df = train_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
train_df = shuffle(train_df)
train_df = train_df.reset_index(drop=True)

valid_df = pd.read_csv(valid_fn)
valid_df.drop(valid_df[valid_df['patch_class'].isin(['Probable Epilation', 'Probable TT', 'Unknown'])].index, inplace = True)
valid_df = valid_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
valid_df = valid_df.replace({'/work/hinashah/data/EGower/': ''}, regex=True)
valid_df = valid_df.reset_index(drop=True)


unique_classes = np.unique(train_df['patch_class'])
unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_df['patch_class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", len(train_df), "Valid size:", len(valid_df))


dg_train = DatasetGenerator(train_df, unique_class_weights, training=True)

def train_generator():
    dg_train.on_epoch_end()
    enqueuer = OrderedEnqueuer(dg_train, use_multiprocessing=True)
    enqueuer.start(workers=16, max_queue_size=512)

    datas = enqueuer.get()

    for idx in range(len(dg_train)):
        yield next(datas)

    enqueuer.stop()


dataset = tf.data.Dataset.from_generator(train_generator,
    output_signature=(tf.TensorSpec(shape = (768, 768, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (3,), dtype = tf.float32),
        tf.TensorSpec(shape = (1,), dtype = tf.float32))
    )

dataset = dataset.batch(64)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
dataset = dataset.prefetch(4)


dg_val = DatasetGenerator(valid_df, unique_class_weights)

def valid_generator():
    enqueuer = OrderedEnqueuer(dg_val, use_multiprocessing=True)
    enqueuer.start(workers=8, max_queue_size=256)
    datas = enqueuer.get()
    for idx in range(len(dg_val)):
        yield next(datas)
    enqueuer.stop()


dataset_validation = tf.data.Dataset.from_generator(valid_generator,
    output_signature=(tf.TensorSpec(shape = (768, 768, 3), dtype = tf.float32), 
        tf.TensorSpec(shape = (3,), dtype = tf.float32),
        tf.TensorSpec(shape = (1,), dtype = tf.float32))
    )

dataset_validation = dataset_validation.batch(128)
options_val = tf.data.Options()
options_val.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset_validation = dataset_validation.with_options(options_val)
dataset_validation = dataset_validation.prefetch(4)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = TTModel()
    model.build(input_shape=(None, 448, 448, 3))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["acc"])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True, 
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=50, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
