import h5py
import tensorflow as tf
import numpy as np

filename = 'data/20181117_Driver1_Trip7.hdf'
tf.keras.backend.set_floatx('float64')
with h5py.File(filename, 'r') as f:
    data_time_serial = np.array(f['/CAN/AccPedal'])  # (8503, 2)
    time_serial = np.expand_dims(data_time_serial.T[0], 0)
    # time_serial = time_serial
    # print(time_serial[250])
    # print(time_serial.shape)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Encoder, self).__init__()
        self.output_dim = output_dim

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.output_dim,
                                       # Return the sequence and state
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')


    def call(self, input_serial, state=None):
        #    The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(input_serial, initial_state=state)

        #    Returns the new sequence and its state.
        return output, state

# length = time_serial.shape[0]
# input = np.empty([1, 200, 1])
# for i in range(length - 200):
#     input = np.concatenate((input, np.reshape(time_serial[i:i+200], [1, 200, 1])))
#
# encoder = Encoder(128)
# # output, state = encoder(np.expand_dims(time_serial, 0))
#
# output, state = encoder(input)
#
# print(output)
# print(state)
