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


class TimeAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(TimeAttention, self).__init__()

    def call(self, x):

        x_t = tf.transpose(x, perm=[0, 2, 1])

        return tf.reduce_sum(tf.matmul(x_t, x), axis=1)
        

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

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[224, 224, 3]), pooling='avg')

        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.noise = tf.keras.layers.GaussianNoise(0.01)
        self.concat = layers.Concatenate(axis=1)
        self.reshape_0 = layers.Reshape((2, 2, 2048))
        self.conv = layers.Conv2D(512, (2, 2))
        self.reshape_1 = layers.Reshape((512,))

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):

        x = self.rescale(x)
        x = self.noise(x)

        q0 = tf.expand_dims(self.resnet(x[:,0:224,0:224,:]), axis=1)
        q1 = tf.expand_dims(self.resnet(x[:,224:,0:224,:]), axis=1)
        q2 = tf.expand_dims(self.resnet(x[:,0:224,224:,:]), axis=1)
        q3 = tf.expand_dims(self.resnet(x[:,224:,224:,:]), axis=1)

        x = self.concat([q0, q1, q2, q3])
        x = self.reshape_0(x)
        x = self.conv(x)
        x = self.reshape_1(x)

        return x

class TTModelPatch(tf.keras.Model):
    def __init__(self):
        super(TTModelPatch, self).__init__()

        self.val_lambda = tf.Variable(initial_value=0.1, trainable=False)
        beta = 0.8
        self.beta_val = tf.constant([beta])# should be between 0.1 and 1.0

        self.features = Features()
        self.P = layers.Dense(3, activation='softmax', name='predictions')
        self.C = layers.Dense(1, activation='sigmoid', name='confidence')
        
    def call(self, x):

        x_f = self.features(x)
        c = self.C(x_f)
        p = self.P(x_f)

        return p, c

    def train_step(self, data):
        x, y_true, sample_weight = data

        with tf.GradientTape() as tape:
            y_pred, confidence = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            y_true = tf.cast(y_true, dtype=tf.float32)
            new_pred = confidence * y_pred + (1.0 - confidence) * y_true
            task_loss = self.compiled_loss(y_true, new_pred, sample_weight=sample_weight, regularization_losses=self.losses)

            confidence_loss = tf.reduce_sum(-tf.math.log(confidence))
            loss = (task_loss + self.val_lambda * confidence_loss)

        self.val_lambda.assign(tf.cond(self.beta_val > confidence_loss, lambda: self.val_lambda / 1.01, lambda:  self.val_lambda / .99))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["confidence_loss"] = confidence_loss
        metrics["val_lambda"] = self.val_lambda
        metrics["task_loss"] = task_loss
        metrics["loss_tt"] = loss
        return metrics


class TTModel(tf.keras.Model):
    def __init__(self, features = None):
        super(TTModel, self).__init__()

        self.val_lambda = tf.Variable(initial_value=0.2, trainable=False)
        beta = 0.8
        self.beta_val = tf.constant([beta])# should be between 0.1 and 1.0

        self.features = Features()
        if(features):
            self.features.resnet = features.resnet
            self.features.conv = features.conv

        self.td = layers.TimeDistributed(self.features)
        self.R = layers.Reshape((-1, 512))

        self.V = layers.Dense(256)
        self.A = Attention(128, 1)
        # self.TA = TimeAttention()
        # self.C = layers.Dense(1, activation='sigmoid', name='confidence')
        self.P = layers.Dense(2, activation='softmax', name='predictions')
        
    def call(self, x):

        x = self.td(x)
        x = self.R(x)

        x_v = self.V(x)
        x_a, w_a = self.A(x, x_v)
        # x_ta = self.TA(x_v)
        # x_a = tf.concat([x_a, x_ta], axis=-1)
        
        # c = self.C(x_a)
        x = self.P(x_a)

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
    #     metrics["loss_c"] = loss
    #     return metrics

class DatasetGenerator:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((None, 448, 448, 3), [2], [1])
            )

        self.dataset = self.dataset.batch(8)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["img"])
            sev = row["class"]

            img_np, head = nrrd.read(img, index_order='C')

            # if len(img_np.shape) == 3:
            #     img_np = np.repeat(np.expand_dims(img_np, axis=0), 4, axis=0)
            #     low = 0.95
            # else:
            num_frames = 4
            img_np = img_np[np.random.choice(16, num_frames, replace=False)]
            low = 0.85

            img_np = np.array([ndimage.rotate(im, np.random.random() * 180 - 90, reshape=False, mode='reflect') for im in img_np])
            img_np = img_np[:,8:-8,8:-8,:]

            zoom_factor = np.random.uniform(low=low, high=1.15)
            img_np = ndimage.zoom(img_np, zoom=[1, zoom_factor, zoom_factor, 1], order=0)

            img_np = tf.image.random_crop(img_np, size=(num_frames, 448, 448, 3))


            img_np = np.array([ndimage.rotate(im, np.random.random() * 180 - 90, reshape=False, mode='reflect') for im in img_np])

            ct = np.random.randint(0, 5)

            if ct == 1:
                saturation_factor = np.random.rand()*10
                img_np = np.array([tf.image.adjust_saturation(im, saturation_factor) for im in img_np])
            elif ct == 2:
                contrast_factor = np.random.rand()*0.5
                img_np = np.array([tf.image.adjust_contrast(im, contrast_factor) for im in img_np])
            elif ct == 3:
                delta = np.random.rand()*0.25
                img_np = np.array([tf.image.adjust_hue(im, delta) for im in img_np])
            elif ct == 4:
                delta = (np.random.rand() - 0.5)*140
                img_np = np.array([tf.image.adjust_brightness(im, delta) for im in img_np])

            yield img_np, tf.one_hot(sev, 2), np.array([self.unique_class_weights[sev]])

class DatasetGeneratorValid:
    def __init__(self, df, unique_class_weights):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32, tf.float32), 
            output_shapes=((None, 448, 448, 3), [2], [1])
            )

        self.dataset = self.dataset.batch(1)
        self.unique_class_weights = unique_class_weights


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["img"])
            sev = row["class"]

            img_np, head = nrrd.read(img, index_order='C')

            # if len(img_np.shape) == 3:
            #     img_np = np.repeat(np.expand_dims(img_np, axis=0), 16, axis=0)

            t, xs, ys, _ = img_np.shape
            xo = (xs - 448)//2
            yo = (ys - 448)//2
            img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]

            # yield img_np, tf.one_hot(sev, 3), np.array([self.unique_class_weights[sev]])
            yield img_np, tf.one_hot(sev, 2), np.array([self.unique_class_weights[sev]])

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



checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_resnet_10122021_weights/stack_training_resnet_10122021"

# csv_path_epi = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_patches_train.csv"
# df_epi = pd.read_csv(csv_path_epi)

# df_epi.drop(df_epi[df_epi['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
# df_epi = df_epi.reset_index()
# df_epi = df_epi.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
# df_epi = df_epi[["image", "patch_class"]]
# df_epi = df_epi.rename(columns={"image": "img", "patch_class": "class"})

csv_path_stacks = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_stacks_16_544_train.csv"
# "/work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_16_544_10082021_train.csv"
 # 

df_stacks = pd.read_csv(csv_path_stacks).replace("/work/jprieto/data/remote/EGower/", "", regex=True)
df_stacks['class'] = (df_stacks['class'] >= 1).astype(int)

# df = pd.concat([df_epi, df_stacks]).reset_index()
df = df_stacks


train_df, valid_df = train_test_split(df, test_size=0.1)

unique_classes = np.unique(train_df['class'])
unique_class_weights = np.array(class_weight.compute_class_weight('balanced', unique_classes, train_df['class']))

print("Unique classes:", unique_classes, unique_class_weights)
print("Train size:", train_df.size, "Valid size:", valid_df.size)

dataset = DatasetGenerator(train_df, unique_class_weights).get()
dataset_validation = DatasetGeneratorValid(valid_df, unique_class_weights).get()

model_patches = TTModelPatch()
model_patches.load_weights(tf.train.latest_checkpoint("/work/jprieto/data/remote/EGower/jprieto/train/patch_training_resnet_11122021_weights/"))

# model = TTModel(model_patches.features)
model = TTModel()
# model.features.trainable = False

model.build(input_shape=(None, None, 448, 448, 3))
model.summary()
optimizer = tf.keras.optimizers.Adam(1e-4)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
# loss_fn = ConfidenceLoss()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["acc"])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    monitor='val_loss',
    mode='auto',
    save_best_only=True, 
    save_weights_only=True)
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(dataset, validation_data=dataset_validation, workers=16, epochs=200, callbacks=[model_early_stopping_callback, model_checkpoint_callback])

