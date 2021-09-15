import tensorflow as tf
from tensorflow.keras import layers

def IoU(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    num_classes = tf.shape(y_true)[-1]

    y_true = tf.reshape(y_true, [-1, num_classes])
    y_pred = tf.reshape(y_pred, [-1, num_classes])
    intersection = 2.0*tf.reduce_sum(y_true * y_pred, axis=0) + 1
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0) + 1.

    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])

    # intersection = 2.0*tf.reduce_sum(y_true * y_pred) + 1
    # union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.

    return 1.0 - tf.reduce_mean(intersection / union)

def make_model():

    drop_prob=0.0

    x0 = tf.keras.Input(shape=[512, 512, 3])

    x = x0
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)
    d0 = x

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)
    d1 = x

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)
    d2 = x

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)
    d3 = x

    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)
    d4 = x

    x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)
    d5 = x

    x = layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)

    x = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)

    x = layers.Concatenate(axis=-1)([x, d5])
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)

    x = layers.Concatenate(axis=-1)([x, d4])
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)

    x = layers.Concatenate(axis=-1)([x, d3])
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)

    x = layers.Concatenate(axis=-1)([x, d2])
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_prob)(x)

    x = layers.Concatenate(axis=-1)([x, d1])
    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Concatenate(axis=-1)([x, d0])
    x = layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', activation='softmax')(x)

    x = layers.Lambda(lambda x: tf.cast(tf.expand_dims(tf.math.argmax(x, axis=-1), axis=-1), dtype=tf.uint8))(x)

    return tf.keras.Model(inputs=x0, outputs=x)


model = make_model()

model_path = "/work/jprieto/data/remote/EGower/jprieto/train/eyes_cropped_resampled_512_seg_train_random_rot"

model.set_weights(tf.keras.models.load_model(model_path, custom_objects={'tf': tf, 'IoU': IoU}).get_weights())

model.summary()

model.save("/work/jprieto/data/remote/EGower/jprieto/trained/eyes_cropped_resampled_512_seg_train_random_rot_09072021.hd5")

