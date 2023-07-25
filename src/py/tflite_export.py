import os
import glob
import sys

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import OrderedEnqueuer

import pandas as pd
import numpy as np
import SimpleITK as sitk


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

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        self.rescale = layers.Rescaling(1/127.5, offset=-1)
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x, training=True):
        x = self.rescale(x)
        x = self.resnet(x)
        x = self.conv(x)
        x = self.avg(x)

        return x


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


class TTModelFeatures(tf.keras.Model):
    def __init__(self):
        super(TTModelFeatures, self).__init__()

        self.features = Features()        
        
    def call(self, x):

        x = self.features(x)

        return x

class TTModelPredict(tf.keras.Model):
    def __init__(self):
        super(TTModelPredict, self).__init__()

        self.V = layers.Dense(256)
        self.A = Attention(128, 1)        
        self.P = layers.Dense(2, activation='softmax', name='predictions')
        
    def call(self, x):

        x_v = self.V(x)
        x_a, x_s = self.A(x, x_v)
        
        x = self.P(x_a)

        return x

checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/train_stack_resnet_31032022_weights/train_stack_resnet_31032022"
out_model_name = "/work/jprieto/data/remote/EGower/jprieto/trained/train_stack_resnet_31032022"

model = TTModel()
model.load_weights(checkpoint_path)
x = tf.keras.layers.Input((None, 448, 448, 3))
model(x)

model_features = TTModelFeatures()
model_features.features = model.features
x = tf.keras.layers.Input((448, 448, 3))
model_features(x)

converter = tf.lite.TFLiteConverter.from_keras_model(model_features)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()



with open(out_model_name + "_features.tflite", 'wb') as f:
  f.write(tflite_model)

model_predict = TTModelPredict()
model_predict.V = model.V
model_predict.A = model.A
model_predict.P = model.P
x = tf.keras.layers.Input((16, 512))
model_predict(x)

converter = tf.lite.TFLiteConverter.from_keras_model(model_predict)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open(out_model_name + "_predict.tflite", 'wb') as f:
  f.write(tflite_model)













































# Convert the model

# saved_model_dir = '/work/jprieto/data/remote/EGower/jprieto/trained/keras_app/VGG19_extract_features_384_avgpool'
# saved_model_dir = '/work/jprieto/data/remote/EGower/jprieto/trained/trachoma_class_input_normals_healthy_sev23_class_features_noflips_normalsvstt_folds/trachoma_class_input_normals_healthy_sev23_class_features_noflips_normalsvstt/trachoma_class_input_normals_healthy_sev23_class_features_noflips_normalsvstt_fold0_train'
# saved_model_dir = '/work/jprieto/data/remote/EGower/jprieto/trained/keras_app/ResNet50_extract_features_384_avgpool'




# saved_model_dir = "/work/jprieto/data/remote/EGower/jprieto/trained/eyes_cropped_resampled_512_seg_train_random_rot_09072021"

# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_model = converter.convert()

# with open(saved_model_dir + ".tflite", 'wb') as f:
#   f.write(tflite_model)


# saved_model_dir = "/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_resnet_att_17012022"


# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_model = converter.convert()

# with open(saved_model_dir + ".tflite", 'wb') as f:
#   f.write(tflite_model)



        

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

#         return context_vector, score

# class Features(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Features, self).__init__()

#         self.resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        
#         self.rescale = layers.Rescaling(1/127.5, offset=-1)
#         self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
#         self.avg = layers.GlobalAveragePooling2D()

#     def compute_output_shape(self, input_shape):
#         return (None, 512)


#     def call(self, x):
        
#         x = self.rescale(x)
#         x = self.resnet(x)
#         x = self.conv(x)
#         x = self.avg(x)

#         return x


# class TTModel(tf.keras.Model):
#     def __init__(self):
#         super(TTModel, self).__init__()

#         self.features = Features()        

#         self.TD = layers.TimeDistributed(self.features)
#         self.R = layers.Reshape((-1, 512))

#         self.V = layers.Dense(256)
#         self.A = Attention(128, 1)        
#         self.P = layers.Dense(2, activation='softmax', name='predictions')
        
#     def call(self, x):

#         x = self.TD(x)
#         x = self.R(x)

#         x_v = self.V(x)
#         x_a, x_s = self.A(x, x_v)
        
#         x = self.P(x_a)
#         x_v_p = self.P(x_v)

#         return x, x_a, x_v, x_s, x_v_p

# class TTModelFeatures(tf.keras.Model):
#     def __init__(self):
#         super(TTModelFeatures, self).__init__()

#         self.features = Features()        
        
#     def call(self, x):

#         x = self.features(x)

#         return x

# class TTModelPredict(tf.keras.Model):
#     def __init__(self):
#         super(TTModelPredict, self).__init__()

#         self.V = layers.Dense(256)
#         self.A = Attention(128, 1)        
#         self.P = layers.Dense(2, activation='softmax', name='predictions')
        
#     def call(self, x):

#         x_v = self.V(x)
#         x_a, x_s = self.A(x, x_v)
        
#         x = self.P(x_a)
#         # x_v_p = self.P(x_v)1

#         return x
#         # return x, x_a, x_v, x_s, x_v_p

# checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_resnet_att_17012022_weights/stack_training_resnet_att_17012022"






