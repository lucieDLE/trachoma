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
# import PIL
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

        self.mobile = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)

        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalMaxPooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):

        x = self.rescale(x)
        x = self.mobile(x)
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

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((448, 448, 3), [3], [1])
            )

        self.dataset = self.dataset.batch(128)
        self.dataset = self.dataset.prefetch(16)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        self.dataset = self.dataset.with_options(options)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["image"])
            patch_class = row["patch_class"]

            img_np = itk.GetArrayViewFromImage(itk.imread(img))

            img_np = ndimage.rotate(img_np, np.random.random() * 180 - 90, reshape=False, mode='reflect')

            zoom_factor = np.random.uniform(low=0.9, high=1.1)
            img_np = ndimage.zoom(img_np, zoom=[zoom_factor, zoom_factor, 1], order=0)

            img_np = tf.image.random_crop(img_np, size=(448, 448, 3))

            ct = np.random.randint(0, 5)

            if ct == 1:
                saturation_factor = np.random.rand()*10
                img_np = tf.image.adjust_saturation(img_np, saturation_factor)
            elif ct == 2:
                contrast_factor = np.random.rand()*0.5
                img_np = tf.image.adjust_contrast(img_np, contrast_factor)
            elif ct == 3:
                delta = np.random.rand()*0.25
                img_np = tf.image.adjust_hue(img_np, delta)
            elif ct == 4:
                delta = (np.random.rand() - 0.5)*140
                img_np = tf.image.adjust_brightness(img_np, delta)

            yield img_np, tf.one_hot(patch_class, 3), np.array([self.unique_class_weights[patch_class]])


# gpus_index = [0]
# print("Using gpus:", gpus_index)
# gpus = tf.config.list_physical_devices('GPU')


# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#         gpu_visible_devices = []
#         for i in gpus_index:
#             gpu_visible_devices.append(gpus[i])
        
#         print(bcolors.OKGREEN, "Using gpus:", gpu_visible_devices, bcolors.ENDC)

#         tf.config.set_visible_devices(gpu_visible_devices, 'GPU')
#         # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(bcolors.FAIL, e, bcolors.ENDC)
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


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = TTModel()
    model.build(input_shape=(None, 448, 448, 3))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["acc"])

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/patch_training_mobile_28122021_weights/patch_training_mobile_28122021"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model_loss_error_callback = LossAndErrorPrintingCallback()

model.fit(dataset, validation_data=dataset_validation, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])
