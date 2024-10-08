
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

import pickle  
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
        
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    
    # grad_model = tf.keras.models.Model(
    #     [model.inputs], [model.features.efficient.get_layer(last_conv_layer_name).output, model.output[0]]
    # )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # last_conv_layer_output, preds = grad_model(img_array)
        preds = model(img_array)[0]        
        last_conv_layer_output = model.features.efficient.get_layer(last_conv_layer_name).output
        if pred_index is None:
            pred_index = tf.argmax(preds, axis=1)[0]
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

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

        self.efficient = tf.keras.applications.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=tf.keras.Input(shape=[448, 448, 3]), pooling=None)
        
        self.center_crop = tf.keras.layers.CenterCrop(448, 448)        
        self.conv = layers.Conv2D(512, (2, 2), strides=(2, 2))
        self.avg = layers.GlobalAveragePooling2D()

    def compute_output_shape(self, input_shape):
        return (None, 512)


    def call(self, x, training=True):

        x = self.center_crop(x)        
        x = self.efficient(x)
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
        x_a, x_s = self.A(x, x_v)
        
        x = self.P(x_a)
        x_v_p = self.P(x_v)

        return x, x_a, x_s, x_v, x_v_p


class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
            
        try:
            row = self.df.loc[idx]
            # img = os.path.join("/work/jprieto/data/remote/EGower/hinashah/Analyses_Set_20220321_Images_stacks/", row["image"].replace(".jpg", ".nrrd"))
            img = os.path.join("hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT_stacks/", row["image"].replace(".jpg", ".nrrd"))
            sev = row["tt sev"]

            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img))

            t, xs, ys, _ = img_np.shape
            xo = (xs - 448)//2
            yo = (ys - 448)//2
            img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]
            
            one_hot = np.zeros(2)
            one_hot[sev] = 1

            return img_np, one_hot
        except Exception as e:
            print(bcolors.FAIL, e, bcolors.ENDC, file=sys.stderr)
            print(bcolors.FAIL, img, bcolors.ENDC, file=sys.stderr)
            return np.zeros([16, 448, 448, 3]), np.zeros(2)



checkpoint_path = "/work/jprieto/data/remote/EGower/jprieto/train/train_stack_efficientv2s_01042022_weights/train_stack_efficientv2s_01042022"

model = TTModel()
model.load_weights(checkpoint_path)
model.build(input_shape=(1, 16, 448, 448, 3))
model.summary()

# csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/trachoma_bsl_mtss_besrat_field_mislabeled.csv"
csv_path_stacks = "hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT.csv"

test_df = pd.read_csv(csv_path_stacks).replace("hinashah/", "", regex=True)
test_df['tt sev'] = (test_df['tt sev'] >= 1).astype(int)


# Remove last layer's softmax

# for l in model.features.efficient.layers:
#     print(l.name)

# print(model.layers[-1])
model.layers[-1].activation = None

# print(model)
# Print what the top predicted class is
# preds = model.predict(img_array)
# print("Predicted:", decode_predictions(preds, top=1)[0])

last_conv_layer_name = "top_activation"

dg_test = DatasetGenerator(test_df)

img, cl = dg_test[0]

print(img.shape)

# for img_array in img:

x_in = tf.keras.layers.Input(shape=(16, 448, 448, 3), dtype=tf.float32)
y = model(x_in, training=False)

print(model.inputs)
print(model.features.efficient.get_layer(last_conv_layer_name).output.shape)
print(model.output)

heatmap = make_gradcam_heatmap(np.expand_dims(img, axis=0), model, last_conv_layer_name)
print(heatmap.shape)

# def test_generator():

#     enqueuer = OrderedEnqueuer(dg_test, use_multiprocessing=True)
#     enqueuer.start(workers=8, max_queue_size=128)

#     datas = enqueuer.get()

#     for idx in range(len(dg_test)):
#         yield next(datas)

#     enqueuer.stop()

# dataset = tf.data.Dataset.from_generator(test_generator,
#     output_signature=(tf.TensorSpec(shape = (None, 448, 448, 3), dtype = tf.float32), 
#         tf.TensorSpec(shape = (2,), dtype = tf.int32))
#     )

# dataset = dataset.batch(1)
# dataset = dataset.prefetch(16)



# dataset_stacks_predict = model.predict(dataset, verbose=True)


# with open(csv_path_stacks.replace(".csv", "_model01042022_prediction.pickle"), 'wb') as f:
#     pickle.dump(dataset_stacks_predict, f)

# print(classification_report(test_df["tt sev"], np.argmax(dataset_stacks_predict[0], axis=1)))