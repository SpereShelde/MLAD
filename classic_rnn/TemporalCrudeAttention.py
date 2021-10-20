from tensorflow import keras
# from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class TemporalCrudeAttention(keras.layers.Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(TemporalCrudeAttention, self).__init__()

    def build(self, input_shape):
        self.temporal_w = self.add_weight(name="temporal_att_weight", shape=(input_shape[2], input_shape[2]),
                                          initializer=keras.initializers.RandomNormal(mean=0., stddev=1.),
                                          trainable=True)
        self.temporal_b = self.add_weight(name="temporal_att_bias", shape=(input_shape[1], input_shape[2]),
                                          initializer="zeros", trainable=True)

        super(TemporalCrudeAttention, self).build(input_shape)

    def call(self, input_time_serials, **kwargs):
        e = K.tanh(K.dot(input_time_serials, self.temporal_w) + self.temporal_b)
        a = K.softmax(e, axis=1)
        output = input_time_serials * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)
