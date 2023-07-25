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

        return context_vector, score

class Features(tf.keras.layers.Layer):
    def __init__(self):
        super(Features, self).__init__()

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[224, 224, 3]), pooling='avg')

        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.concat = layers.Concatenate(axis=1)
        self.reshape_0 = layers.Reshape((2, 2, 2048))
        self.conv = layers.Conv2D(512, (2, 2))
        self.reshape_1 = layers.Reshape((512,))

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x):

        x = self.rescale(x)

        q0 = tf.expand_dims(self.resnet(x[:,0:224,0:224,:]), axis=1)
        q1 = tf.expand_dims(self.resnet(x[:,224:,0:224,:]), axis=1)
        q2 = tf.expand_dims(self.resnet(x[:,0:224,224:,:]), axis=1)
        q3 = tf.expand_dims(self.resnet(x[:,224:,224:,:]), axis=1)

        x = self.concat([q0, q1, q2, q3])
        x = self.reshape_0(x)
        x = self.conv(x)
        x = self.reshape_1(x)

        return x


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
        x_a, w_s = self.A(x, x_v)
        # x_ta = self.TA(x_v)
        # x_a = tf.concat([x_a, x_ta], axis=-1)
        
        # c = self.C(x_a)
        x = self.P(x_a)
        x_v_p = self.P(x_v)

        return x, x_a, x_v, w_s, x_v_p

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
    def __init__(self, df):
        self.df = df

        self.dataset = tf.data.Dataset.from_generator(self.generator,
            output_types=(tf.float32, tf.int32), 
            output_shapes=((None, 448, 448, 3), [2])
            )

        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(128)


    def get(self):
        return self.dataset
    
    def generator(self):

        for idx, row in self.df.iterrows():
            
            img = os.path.join("/work/jprieto/data/remote/EGower/", row["img"])
            sev = row["class"]

            # img_np = itk.GetArrayViewFromImage(itk.imread(img))
            img_np, head = nrrd.read(img, index_order='C')

            if len(img_np.shape) == 3:
                img_np = np.expand_dims(img_np, axis=0)

            t, xs, ys, _ = img_np.shape
            xo = (xs - 448)//2
            yo = (ys - 448)//2
            img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]

            yield img_np, tf.one_hot(sev, 2)

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



# checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/patch_stack_resnet_att_29112021_weights/patch_stack_resnet_att_29112021.ckpt"

# csv_path_epi = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_patches_test.csv"
# df_epi = pd.read_csv(csv_path_epi)

# df_epi.drop(df_epi[df_epi['patch_class'].isin(['Probable Epilation', 'Probable TT'])].index, inplace = True)
# df_epi = df_epi.reset_index()
# df_epi = df_epi.replace({'Healthy': 0, 'TT': 1, 'Epilation': 2})
# df_epi = df_epi[["image", "patch_class"]]
# df_epi = df_epi.rename(columns={"image": "img", "patch_class": "class"})

csv_path_stacks = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_11012021/trachoma_normals_healthy_sev123_epi_stacks_16_544_test.csv"

df_stacks = pd.read_csv(csv_path_stacks)
df_stacks['class'] = (df_stacks['class'] >= 1).astype(int)

dataset_stacks = DatasetGenerator(df_stacks).get()

# model = TTModel()

# model.load_weights(checkpoint_path)

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_resnet_10122021_weights/stack_training_resnet_10122021"

# model = tf.keras.models.load_model(model_path, custom_objects={'TTModel': TTModel})
model = TTModel()
model.load_weights(checkpoint_path)




    
dataset_stacks_predict = model.predict(dataset_stacks, verbose=True)

output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"


with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_epi_stacks_16_544_test_10122021.pickle'), 'wb') as f:
    pickle.dump(dataset_stacks_predict, f)





csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_16_544_10082021_test.csv"

df_stacks = pd.read_csv(csv_path_stacks)
df_stacks['class'] = (df_stacks['class'] >= 1).astype(int)

dataset_stacks = DatasetGenerator(df_stacks).get()

dataset_stacks_predict = model.predict(dataset_stacks, workers=8, verbose=True)
output_dir = "/work/jprieto/data/remote/EGower/jprieto/test_output/"

with open(os.path.join(output_dir, 'trachoma_normals_healthy_sev123_05182021_stack_16_544_10082021_test_10122021.pickle'), 'wb') as f:
    pickle.dump(dataset_stacks_predict, f)
