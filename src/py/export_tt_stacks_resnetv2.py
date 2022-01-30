
import tensorflow as tf
from tensorflow.keras import layers

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

class TTModel(tf.keras.Model):
    def __init__(self):
        super(TTModel, self).__init__()

        x0 = tf.keras.Input(shape=[384, 384, 3])
        self.resnet50 = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=x0, pooling='avg')
        
        self.PreProcess = tf.keras.layers.Rescaling(1/127.5, offset=-1)
        self.TD = layers.TimeDistributed(self.resnet50)

        x0 = tf.keras.Input(shape=[None, 2048])
        
        self.V = layers.Dense(256)

        self.A = Attention(128, 256)
        self.P = layers.Dense(1, activation='sigmoid', name='predictions')
        
        v = self.V(x0)
        x, w_a = self.A(x0, v)
        x = self.P(x)

        self.model_prediction = tf.keras.Model(inputs=x0, outputs=x)

    def call(self, x):

        x = self.PreProcess(x)
        x = self.TD(x)

        x = self.model_prediction(x)

model = tf.keras.models.load_model("/work/jprieto/data/remote/EGower/jprieto/train/stack_training_09292021_64_448_2class/", custom_objects={"TTModel": TTModel, 'Attention': Attention})
model.summary()

# x0 = tf.keras.Input(shape=[32, 384, 384, 3])
# x = model.TD(x0)
# q = model.Q(x)
# v = model.V(x)
# x, w_a = model.A(q, v)
# x = model.P(x)
# full_model = tf.keras.Model(inputs=x0, outputs=x)
# full_model.summary()
# full_model.save("/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_2class")

x0 = tf.keras.Input(shape=[384, 384, 3])

resnet50 = model.resnet50
resnet50.summary()
resnet50.save("/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_2class_resnet50")

x0 = tf.keras.Input(shape=[32, 2048])
q = model.Q(x0)
v = model.V(x0)
x, w_a = model.A(q, v)
x = model.P(x)
model_prediction = tf.keras.Model(inputs=x0, outputs=x)
model_prediction.summary()
model_prediction.save("/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_2class_prediction")


#Export to tflite

converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
tflite_model = converter.convert()
with open('/work/jprieto/data/remote/EGower/jprieto/trained/full.tflite', 'wb') as f:
  f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(resnet50)
tflite_model = converter.convert()
with open('/work/jprieto/data/remote/EGower/jprieto/trained/features.tflite', 'wb') as f:
  f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model_prediction)
tflite_model = converter.convert()
with open('/work/jprieto/data/remote/EGower/jprieto/trained/prediction.tflite', 'wb') as f:
  f.write(tflite_model)