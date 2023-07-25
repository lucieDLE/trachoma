from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend
# from keras.utils import control_flow_util
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
        self.random_zoom = tf.keras.layers.RandomZoom((-0.3, 0.3))
        # self.resize = tf.keras.layers.Resizing(480, 480)
        self.center_crop0 = tf.keras.layers.CenterCrop(512, 512)
        self.random_crop = tf.keras.layers.RandomCrop(448, 448)
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
        # x = tf.cond(tf.cast(training, tf.bool), 
        #     lambda: self.resize(x),
        #     lambda: x)
        x = self.center_crop(x)
        x = self.random_crop(x)
        x = self.center_crop(x)
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

        return x

    # def train_step(self, data):
    #     x, y_true, sample_weight = data

    #     with tf.GradientTape() as tape:
    #         y_pred, confidence = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         y_true = tf.cast(y_true, dtype=tf.float32)
    #         new_pred = confidence * y_pred + (1.0 - confidence) * y_true
    #         task_loss = self.compiled_loss(y_true, new_pred, sample_weight=sample_weight, regularization_losses=self.losses)

    #         confidence_loss = tf.reduce_sum(-tf.math.log(confidence))
    #         loss = (task_loss + self.val_lambda * confidence_loss)

    #     self.val_lambda.assign(tf.cond(self.beta_val > confidence_loss, lambda: self.val_lambda / 1.01, lambda:  self.val_lambda / .99))

    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     # Update metrics (includes the metric that tracks the loss)
    #     self.compiled_metrics.update_state(y_true, y_pred)
    #     # Return a dict mapping metric names to current value
    #     metrics = {m.name: m.result() for m in self.metrics}
    #     metrics["confidence_loss"] = confidence_loss
    #     metrics["val_lambda"] = self.val_lambda
    #     metrics["task_loss"] = task_loss
    #     metrics["loss_tt"] = loss
    #     return metrics

class DatasetGenerator:
    def __init__(self, df, unique_class_weights):
        self.df = df        
        self.unique_class_weights = unique_class_weights


    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):

        row = self.df.loc[idx]
            
        img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
        patch_class = row["patch_class"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

        xs, ys, _ = img_np.shape
        xo = (xs - 768)//2
        yo = (ys - 768)//2
        img_np = img_np[xo:xo + 768, yo:yo + 768,:]

        # img_np = ndimage.rotate(img_np, np.random.random() * 180 - 90, reshape=False, mode='reflect')

        # zoom_factor = np.random.uniform(low=0.9, high=1.1)
        # img_np = ndimage.zoom(img_np, zoom=[zoom_factor, zoom_factor, 1], order=0)

        # img_np = tf.image.random_crop(img_np, size=(448, 448, 3))

        return img_np, tf.one_hot(patch_class, 3), np.array([self.unique_class_weights[patch_class]])


train_df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_01132022/trachoma_normals_healthy_sev123_epi_patches_train_train.csv")
train_df.drop(train_df[train_df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
train_df = train_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
train_df = shuffle(train_df)
train_df = train_df.reset_index(drop=True)

valid_df = pd.read_csv("/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_01132022/trachoma_normals_healthy_sev123_epi_patches_train_valid.csv")
valid_df.drop(valid_df[valid_df['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
valid_df = valid_df.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
valid_df = valid_df.reset_index(drop=True)


unique_classes = np.unique(train_df['patch_class'])
unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_df['patch_class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", len(train_df), "Valid size:", len(valid_df))


dg_train = DatasetGenerator(train_df, unique_class_weights)

def train_generator():
    # dg_train.on_epoch_end()
    # enqueuer = OrderedEnqueuer(dg_train, use_multiprocessing=True)
    # enqueuer.start(workers=12, max_queue_size=24)

    # datas = enqueuer.get()

    # for data in datas:
    #     yield data

    # enqueuer.stop()

    for idx in range(len(dg_train)):
        yield dg_train[idx]


dataset = tf.data.Dataset.from_generator(train_generator,
    output_types=(tf.float32, tf.int32, tf.float32), 
    output_shapes=((768, 768, 3), [3], [1]))

dataset = dataset.batch(128)
options = tf.data.Options()


options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options.threading.private_threadpool_size = 24

dataset = dataset.with_options(options)
dataset = dataset.prefetch(4)


dg_val = DatasetGenerator(valid_df, unique_class_weights)

def valid_generator():    
    # enqueuer = OrderedEnqueuer(dg_val, use_multiprocessing=True)
    # enqueuer.start(workers=12, max_queue_size=24)

    # datas = enqueuer.get()

    # for data in datas:
    #     yield data

    # enqueuer.stop()

    for idx in range(len(dg_val)):
        yield dg_val[idx]


dataset_validation = tf.data.Dataset.from_generator(valid_generator,
    output_types=(tf.float32, tf.int32, tf.float32), 
    output_shapes=((768, 768, 3), [3], [1]))

dataset_validation = dataset_validation.batch(1)
options_val = tf.data.Options()
options_val.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
options_val.threading.private_threadpool_size = 24
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

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/patch_training_resnet_16012022_weights/patch_training_resnet_16012022"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback], 
    workers=24,
    use_multiprocessing=True)
