import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input

import pandas as pd
import numpy as np
import itk
import pickle

# python source/trachoma/src/py/predict.py --csv /work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_32_384_test.csv --prediction_type class --out /work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_32_384_test_09232021_prediction.csv --image_dimension 3 --model /work/jprieto/data/remote/EGower/jprieto/trained/stack_training_09232021_2class

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

        return context_vector, attention_weights, score

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        x0 = tf.keras.Input(shape=[384, 384, 3])
        self.resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x0, pooling='avg')
        
        self.TD = layers.TimeDistributed(self.resnet50, input_shape=[32, 384, 384, 3])

        self.Q = layers.Dense(128)
        self.V = layers.Dense(256)

        self.A = Attention(64, 256)
        self.P = layers.Dense(1, activation='sigmoid', name='predictions')

    def call(self, x):

        x = preprocess_input(x)
        x_f = self.TD(x)

        q = self.Q(x_f)
        v = self.V(x_f)

        x_v, w_a, score = self.A(q, v)

        x = self.P(x_v)

        return x, x_v, v, w_a, score, x_f

class TTModelFeatures(tf.keras.Model):
    def __init__(self):
        super(TTModelFeatures, self).__init__()

        self.Q = layers.Dense(128)
        self.V = layers.Dense(256)

        self.A = Attention(64, 256)
        self.P = layers.Dense(1, activation='sigmoid', name='predictions')

    def call(self, x):

        q = self.Q(x)
        v = self.V(x)

        x_v, w_a, score = self.A(q, v)

        x = self.P(x_v)

        return x

checkpoint_dir = "/work/jprieto/data/remote/EGower/jprieto/train/stack_training_09232021_64_448_2class/"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = TTModel()
model.load_weights(latest)

model_features = TTModelFeatures()
model_features.Q = model.Q
model_features.V = model.V
model_features.A = model.A
model_features.P = model.P

fname = "/work/jprieto/data/remote/EGower/jprieto/trachoma_normals_healthy_sev123_05182021_stack_32_384_test.csv"
out_fname = fname.replace('.csv', '_10042021.csv')

df = pd.read_csv(fname)

predictions = []
values = []
frames_values = []
scores = []
frames_prediction = []


for idx, row in df.iterrows():
    img = itk.GetArrayViewFromImage(itk.imread(row['img']))
    img = np.expand_dims(img, 0)
    x, x_v, x_vf, w_a, score, x_feat = model.predict(img)

    predictions.append(np.round(x[0][0]))
    values.append(x_v[0])
    frames_values.append(x_vf[0])
    scores.append(score[0])

    fr_p = []
    x_feat = x_feat.reshape(-1, 1, 1, 2048)
    for frame_f in x_feat:
        x_f = model_features(frame_f)
        fr_p.append(np.round(x_f[0]).reshape(-1))
    frames_prediction.append(np.array(fr_p).reshape(-1))


df["prediction"] = predictions

df.to_csv(out_fname.replace('.csv', '_prediction.csv'), index=False)

with open(out_fname.replace(".csv", "_values.pickle"), 'wb') as f:
    pickle.dump(values, f)

with open(out_fname.replace(".csv", "_frames_values.pickle"), 'wb') as f:
    pickle.dump(frames_values, f)

with open(out_fname.replace(".csv", "_frames_prediction.pickle"), 'wb') as f:
    pickle.dump(frames_prediction, f)

with open(out_fname.replace(".csv", "_scores.pickle"), 'wb') as f:
    pickle.dump(scores, f)