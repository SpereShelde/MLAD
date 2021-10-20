from tensorflow import keras
# from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class OriginalLSTMDualAttention(keras.layers.Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(OriginalLSTMDualAttention, self).__init__()

    def build(self, input_shape):
        self.spatial_w = self.add_weight(name="spatial_att_weight", shape=(input_shape[2], 1),
                                         initializer=keras.initializers.RandomNormal(mean=0., stddev=1.),
                                         trainable=True)
        self.spatial_b = self.add_weight(name="spatial_att_bias", shape=(input_shape[1], 1),
                                         initializer="zeros", trainable=True)

        self.temporal_w = self.add_weight(name="temporal_att_weight", shape=(input_shape[2], input_shape[2]),
                                          initializer=keras.initializers.RandomNormal(mean=0., stddev=1.),
                                          trainable=True)
        self.temporal_b = self.add_weight(name="temporal_att_bias", shape=(input_shape[1], input_shape[2]),
                                          initializer="zeros", trainable=True)

        super(OriginalLSTMDualAttention, self).build(input_shape)

    def call(self, input_time_serials):
        e1 = K.tanh(K.dot(input_time_serials, self.temporal_w) + self.temporal_b)
        a1 = K.softmax(e1, axis=1)
        attention_time_serials = input_time_serials * a1
        e2 = K.tanh(K.dot(attention_time_serials, self.spatial_w) + self.temporal_b)
        a2 = K.softmax(e2, axis=2)
        output = attention_time_serials * a2

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)
