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
    def __init__(self, df, training=False):
        self.df = df        
        self.training=training

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.loc[idx]            
        img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
        patch_class = row["patch_class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))
        
        if(self.training):
            sigma = np.random.uniform(high=5.0)            
            img_np = ndimage.gaussian_filter(img_np, sigma=(sigma, sigma, 0.0))

        xs, ys, _ = img_np.shape
        xo = (xs - 448)//2
        yo = (ys - 448)//2

        img_np = img_np[xo:xo + 448, yo:yo + 448,:]

        one_hot = np.zeros(3)
        one_hot[patch_class] = 1

        return img_np, one_hot

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)




for fold in range(2, 3):

    print(bcolors.OKBLUE, "Testing fold", fold, bcolors.ENDC)
    checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/train_folds_effnetv2s/train_patch_efficientv2s_30042022_weights_fold_" + str(fold) + "/train_patch_efficientv2s_30042022" 
    

    model = TTModel()
    model.load_weights(checkpoint_path)
    
    csv_path = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_20220326/trachoma_bsl_mtss_besrat_field_patches_train_20220326_trainfold" + str(fold) + "_test.csv"
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


    with open(csv_path.replace(".csv", "_fold" + str(fold) + "_30042022_prediction.pickle"), 'wb') as f:
        pickle.dump(dataset_predict, f)

    print(classification_report(test_df["patch_class"], np.argmax(dataset_predict[0], axis=1)))
