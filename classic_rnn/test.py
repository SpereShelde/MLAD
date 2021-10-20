import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

y_true = tf.cast([
    [
        [1., 1., 2., 2.]
    ],
    [
        [1., 1., 2., 3.]
    ],
], tf.float32)
y_pred = tf.cast([
    [-1., -1., -2., -2.],
    [1., 1., 2., 3.],
], tf.float32)

y_true_2d = tf.reshape(y_true, [y_true.shape[0], y_true.shape[2]])
# print(y_true_2d.shape)
cos_loss = tf.keras.losses.cosine_similarity(y_true_2d, y_pred, axis=1)
print(cos_loss)
sub = tf.subtract(y_true_2d, y_pred)
# print(sub)
ae_loss = tf.reduce_mean(tf.abs(tf.subtract(y_true_2d, y_pred)), axis=1)
print(ae_loss)
loss = ae_loss + cos_loss
print(loss)


# a = tf.ones([1,100],tf.int32)
# reduce_m = tf.math.reduce_mean(a)
# print(reduce_m)
