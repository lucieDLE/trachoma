
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input

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
        super(TTModel, self).__init__(name='')

        x0 = tf.keras.Input(shape=[384, 384, 3])
        self.resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x0, pooling='avg')        
        self.td = layers.TimeDistributed(self.resnet50)

        self.q = layers.Dense(128)
        self.v = layers.Dense(256)

        self.att = Attention(128, 256)
        self.pred = layers.Dense(1, activation='sigmoid', name='predictions')

    def call(self, x0, training=False):

        x = preprocess_input(x0)

        x = self.td(x)

        q = self.q(x)
        v = self.v(x)

        x = self.att(q, v)

        return self.pred(x)

def make_model():
    
    x0 = tf.keras.Input(shape=[384, 384, 3])
    resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x0, pooling='avg')

    x0 = tf.keras.Input(shape=[None, 384, 384, 3])
    x = preprocess_input(x0)
    x = layers.TimeDistributed(resnet50)(x)

    q = layers.Dense(128)(x)
    v = layers.Dense(256)(x)

    x, w_a = Attention(64, 256)(q, v)

    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    return tf.keras.Model(inputs=x0, outputs=x)


model = make_model()

model_path = "/work/jprieto/data/remote/EGower/train/stack_training_09072021_64_448_2class_09102021"

model.set_weights(tf.keras.models.load_model(model_path, custom_objects={'tf': tf, 'Attention': Attention}).get_weights())

model.summary()

model.save("/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_09072021_2class")


x0 = tf.keras.Input(shape=[384, 384, 3])
x = preprocess_input(x0)
resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
td = tf.keras.layers.TimeDistributed(resnet50)

td.set_weights(model.layers[3].get_weights())
resnet50.summary()
resnet50.save("/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_09072021_2class_resnet50")

x0 = tf.keras.Input(shape=[32, 2048])
q = model.layers[4](x0)
v = model.layers[5](x0)
x, w_a = model.layers[6](q, v)
x = model.layers[7](x)
feat_model = tf.keras.Model(inputs=x0, outputs=x)
feat_model.summary()
feat_model.save("/work/jprieto/data/remote/EGower/jprieto/trained/stack_training_09072021_2class_prediction")