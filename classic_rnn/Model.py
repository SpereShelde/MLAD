import math
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from classic_rnn.TemporalCrudeAttention import TemporalCrudeAttention
import os
from DatasetReader import DatasetReader
from keras_multi_head import MultiHead

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
# os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
tf.keras.backend.set_floatx('float64')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def compare(item1, item2):
    r1 = abs(item1[0] - item1[1])
    r2 = abs(item2[0] - item2[1])
    if r1 > r2:
        return -1
    elif r1 < r2:
        return 1
    else:
        return 0

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.subtract(tf.reshape(y_true, [-1, y_true.shape[2]]), y_pred)), axis=1)

def prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows = 8, save=False, show=True):
    plt.figure(figsize=(30, 16))
    for i in range(rows*rows):
        idx = random.randint(0, inputs_train.shape[0])
        plt.subplot(rows, rows, i + 1)
        color = 'purple'
        if sample_weights[idx] == 1:
            color = 'y'
        elif sample_weights[idx] == 3:
            color = 'b'
        elif sample_weights[idx] == 6:
            color = 'g'
        for j in range(feature_size):
            plt.plot(np.arange(monitor_window_length), inputs_train[idx, :, j], color=color)
            plt.scatter(monitor_window_length + target_skip_steps - 1, targets_train[idx, 0, j], color='r')
        plt.ylim([0, 1])
        plt.xlim([0, monitor_window_length + target_skip_steps])
    if save:
        for i in range(1, 20):
            file_name = f"input-map-fs{feature_size}-{i}.png"
            if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
                plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
                break
    if show:
        plt.show()

# def uni_time_serial_lstm():
#     print("uni time serial lstm")
#     data_reader = DatasetReader(
#         ["20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
#          "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf"])
#     inputs_train, targets_train, inputs_test, targets_test = data_reader.sample(['CAN/VehicleSpeed'],
#                                                                                 time_steps=monitor_window_length,
#                                                                                 sample_interval=window_sample_interval,
#                                                                                 target_sequence=False,
#                                                                                 ignore_all_zero=False,
#                                                                                 test_on_file=True)
#
#     print("train input shape", inputs_train.shape)
#     print("train target shape", inputs_train.shape)
#     print("test input shape", inputs_test.shape)
#     print("test target shape", targets_test.shape)
#
#     model = keras.Sequential()
#     model.add(layers.LSTM(256, dropout=0.02, recurrent_dropout=0.02, return_sequences=False))
#     model.add(layers.Dense(1))
#     model.build(input_shape=(batch_size, monitor_window_length, 1))
#     model.compile(optimizer='sgd', loss='mse')
#     # model.summary()
#     # exit(0)
#
#     model.fit(inputs_train, targets_train, batch_size=batch_size, epochs=epochs_num, shuffle=True)
#
#     res = model.evaluate(inputs_test, targets_test)
#     print(res)
#
#     x = np.arange(plot_length)
#     outputs = model(inputs_test)
#
#     flat_results = list(zip(np.array(outputs).reshape(-1), np.array(targets_test).reshape(-1)))
#     flat_results.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
#
#     sorted_outputs = list(map(list, zip(*flat_results[:plot_length])))[0]
#     sorted_targets = list(map(list, zip(*flat_results[:plot_length])))[1]
#
#     plt.scatter(x, np.array(sorted_outputs).reshape(-1), label='predict', c='r', marker='+')
#     plt.scatter(x, np.array(sorted_targets).reshape(-1), label='target', c='b', marker='x')
#
#     # plt.xlabel('idx')
#     # plt.ylabel('value')
#     plt.title('Uni Time Serial LSTM')
#     # plt.legend()
#
#     for i in range(1, 10):
#         file_name = f"uni-lstm-ws{monitor_window_length}-bs{batch_size}-epo{epochs_num}-res{res}-{i}.png"
#         if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
#             plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
#             # plt.show()
#             break


def multi_time_serial_lstm(monitor_window_length, window_sample_interval, target_skip_steps, batch_size, epochs_num, feature_names):
    data_reader = DatasetReader(
        [
            "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
            "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
            "20181117_Driver1_Trip7.hdf", "20181117_Driver1_Trip8.hdf", "20181203_Driver1_Trip9.hdf",
            "20181203_Driver1_Trip10.hdf",
        ])
    inputs_train, targets_train, inputs_test, targets_test, feature_names, normalized_test_data_time_serials = data_reader.sample(feature_names,
                                                                                               time_steps=monitor_window_length,
                                                                                               sample_interval=window_sample_interval,
                                                                                               target_sequence=False,
                                                                                               ignore_all_zero=False,
                                                                                               test_on_file=True,
                                                                                               target_skip_steps=target_skip_steps,
                                                                                               test_files=["20181117_Driver1_Trip7.hdf"])
    print("train input shape", inputs_train.shape)
    print("train target shape", targets_train.shape)
    print("test input shape", inputs_test.shape)
    print("test target shape", targets_test.shape)

    feature_size = inputs_test.shape[2]

    inputs_train_diff_min = np.min(np.max(inputs_train, axis=1) - np.min(inputs_train, axis=1), axis=1)
    sample_weights = np.select([inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 3, 6])

    # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)

    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(256, dropout=0.05, return_sequences=False)))
    model.add(layers.Dense(64))
    model.add(layers.Dense(16))
    model.add(layers.Dense(feature_size))
    model.build(input_shape=(batch_size, monitor_window_length, feature_size))
    model.compile(optimizer='adam', loss=loss_fn)

    # model.summary()
    # exit(0)

    model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)

    res = model.evaluate(inputs_test, targets_test)
    print(res)

    plot_length = min(4000, ((inputs_test.shape[0] - monitor_window_length - target_skip_steps) // 100) * 100)
    x = np.arange(plot_length).reshape(-1, 1)
    inputs_test = inputs_test[:plot_length]
    outputs = model(inputs_test)

    plt.figure(figsize=(30, 16))
    plt.suptitle(f'{feature_size} Features LSTM - {monitor_window_length} TimeSteps - Jump {target_skip_steps} Steps - {window_sample_interval}StepInterval-{batch_size}Batch-{epochs_num}Epochs')

    rows = min(4, feature_size)
    for i in range(feature_size):
        plt.subplot(rows, math.ceil(feature_size/rows), i + 1)
        plt.plot(x, np.array(outputs[:, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, np.array(normalized_test_data_time_serials[monitor_window_length+target_skip_steps:monitor_window_length+target_skip_steps+plot_length, i]).reshape(-1, 1), label='target', c='b', marker='.')
        plt.title(f'{feature_names[i]}')

    # sorted_outputs = np.array(list(map(list, zip(*flat_results[:plot_length])))[0])
    # sorted_targets = np.array(list(map(list, zip(*flat_results[:plot_length])))[1])
    # for i in range(feature_size):
    #     plt.subplot(4, math.ceil(feature_size / 2), feature_size + i + 1)
    #     plt.scatter(x, sorted_outputs[:, i].reshape(1, -1), label='predict', c='r', marker='+')
    #     plt.scatter(x, sorted_targets[:, 0, i].reshape(1, -1), label='target', c='b', marker='x')
    #     plt.title(f'{feature_names[i]}')
    for i in range(1, 10):
        file_name = f"mul-lstm-res{res}-{i}.png"
        if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
            plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
            # plt.show()
            break

def multi_time_serial_lstm_crude_attention(monitor_window_length, window_sample_interval, target_skip_steps, batch_size, epochs_num, feature_names):
    data_reader = DatasetReader(
        [
            "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
            "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
            "20181117_Driver1_Trip7.hdf", "20181117_Driver1_Trip8.hdf", "20181203_Driver1_Trip9.hdf",
            "20181203_Driver1_Trip10.hdf",
        ])
    # feature_names = []
    # feature_names = ['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X']
    inputs_train, targets_train, inputs_test, targets_test, feature_names, normalized_test_data_time_serials = data_reader.sample(feature_names,
                                                                                               time_steps=monitor_window_length,
                                                                                               sample_interval=window_sample_interval,
                                                                                               target_sequence=False,
                                                                                               ignore_all_zero=False,
                                                                                               test_on_file=True,
                                                                                               target_skip_steps=target_skip_steps,
                                                                                               test_files=["20181117_Driver1_Trip7.hdf"])
    print("train input shape", inputs_train.shape)
    print("train target shape", targets_train.shape)
    print("test input shape", inputs_test.shape)
    print("test target shape", targets_test.shape)

    feature_size = inputs_test.shape[2]

    inputs_train_diff_min = np.min(np.max(inputs_train, axis=1) - np.min(inputs_train, axis=1), axis=1)
    sample_weights = np.select([inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 3, 6])

    # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)

    model = keras.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(128, dropout=0.05, return_sequences=False)))
    model.add(TemporalCrudeAttention(return_sequences=True))
    model.add(layers.Dense(32))
    model.add(layers.Dense(feature_size))
    model.build(input_shape=(batch_size, monitor_window_length, feature_size))
    model.compile(optimizer='adam', loss=loss_fn)

    # model.summary()
    # exit(0)

    model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)

    res = model.evaluate(inputs_test, targets_test)
    print(res)

    plot_length = min(4000, ((inputs_test.shape[0] - monitor_window_length - target_skip_steps) // 100) * 100)
    x = np.arange(plot_length).reshape(-1, 1)
    inputs_test = inputs_test[:plot_length]
    outputs = model(inputs_test)

    plt.figure(figsize=(30, 16))
    plt.suptitle(f'{feature_size} Features LSTM Crude Attn - {monitor_window_length} TimeSteps - Jump {target_skip_steps} Steps - {window_sample_interval}StepInterval-{batch_size}Batch-{epochs_num}Epochs')

    rows = min(4, feature_size)
    for i in range(feature_size):
        plt.subplot(rows, math.ceil(feature_size/rows), i + 1)
        plt.plot(x, np.array(outputs[:, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, np.array(normalized_test_data_time_serials[monitor_window_length+target_skip_steps:monitor_window_length+target_skip_steps+plot_length, i]).reshape(-1, 1), label='target', c='b', marker='.')
        plt.title(f'{feature_names[i]}')

    # sorted_outputs = np.array(list(map(list, zip(*flat_results[:plot_length])))[0])
    # sorted_targets = np.array(list(map(list, zip(*flat_results[:plot_length])))[1])
    # for i in range(feature_size):
    #     plt.subplot(4, math.ceil(feature_size / 2), feature_size + i + 1)
    #     plt.scatter(x, sorted_outputs[:, i].reshape(1, -1), label='predict', c='r', marker='+')
    #     plt.scatter(x, sorted_targets[:, 0, i].reshape(1, -1), label='target', c='b', marker='x')
    #     plt.title(f'{feature_names[i]}')
    for i in range(1, 10):
        file_name = f"mul-lstm-crude-attention-res{res}-{i}.png"
        if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
            plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
            # plt.show()
            break

def multi_time_serial_lstm_transformer_attention(monitor_window_length, window_sample_interval, target_skip_steps, batch_size, epochs_num, feature_names):
    data_reader = DatasetReader(
        [
            "20181113_Driver1_Trip1.hdf", "20181113_Driver1_Trip2.hdf", "20181116_Driver1_Trip3.hdf",
            "20181116_Driver1_Trip4.hdf", "20181116_Driver1_Trip5.hdf", "20181116_Driver1_Trip6.hdf",
            "20181117_Driver1_Trip7.hdf", "20181117_Driver1_Trip8.hdf", "20181203_Driver1_Trip9.hdf",
            "20181203_Driver1_Trip10.hdf",
        ])
    inputs_train, targets_train, inputs_test, targets_test, feature_names, normalized_test_data_time_serials = data_reader.sample(feature_names,
                                                                                               time_steps=monitor_window_length,
                                                                                               sample_interval=window_sample_interval,
                                                                                               target_sequence=False,
                                                                                               ignore_all_zero=False,
                                                                                               test_on_file=True,
                                                                                               target_skip_steps=target_skip_steps,
                                                                                               test_files=["20181117_Driver1_Trip7.hdf"])
    print("train input shape", inputs_train.shape)
    print("train target shape", targets_train.shape)
    print("test input shape", inputs_test.shape)
    print("test target shape", targets_test.shape)

    feature_size = inputs_test.shape[2]

    inputs_train_diff_min = np.min(np.max(inputs_train, axis=1) - np.min(inputs_train, axis=1), axis=1)
    sample_weights = np.select([inputs_train_diff_min < 0.05, inputs_train_diff_min < 0.1, inputs_train_diff_min >= 0.1], [1, 3, 6])

    # prin_input(inputs_train, targets_train, sample_weights, feature_size, monitor_window_length, target_skip_steps, rows=8, save=False, show=True)

    model = keras.Sequential()
    model.add(MultiHead(layers.Bidirectional(layers.LSTM(128, dropout=0.05, return_sequences=False)), layer_num=5, name='Multi-LSTMs'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(feature_size))
    model.build(input_shape=(batch_size, monitor_window_length, feature_size))
    model.compile(optimizer='adam', loss=loss_fn)

    # model.summary()
    # exit(0)

    model.fit(inputs_train, targets_train, validation_split=0.2, batch_size=batch_size, epochs=epochs_num, shuffle=True, sample_weight=sample_weights)

    res = model.evaluate(inputs_test, targets_test)
    print(res)

    plot_length = min(4000, ((inputs_test.shape[0] - monitor_window_length - target_skip_steps) // 100) * 100)

    ploted = 0
    outputs = []
    while ploted < plot_length:
        to_plot = min(plot_length, ploted+1000)
        outputs.append(model(inputs_test[ploted:to_plot,:,:]))
        ploted += 1000
    x = np.arange(plot_length).reshape(-1, 1)
    outputs = np.vstack(outputs)

    # x = np.arange(plot_length).reshape(-1, 1)
    # inputs_test = inputs_test[:plot_length]
    # outputs = model(inputs_test)

    plt.figure(figsize=(30, 16))
    plt.suptitle(f'{feature_size} Features LSTM Transformer Attn - {monitor_window_length} TimeSteps - Jump {target_skip_steps} Steps - {window_sample_interval}StepInterval-{batch_size}Batch-{epochs_num}Epochs')

    rows = min(4, feature_size)
    for i in range(feature_size):
        plt.subplot(rows, math.ceil(feature_size/rows), i + 1)
        plt.plot(x, np.array(outputs[:, i]).reshape(-1, 1), label='predict', c='r', marker='.')
        plt.plot(x, np.array(normalized_test_data_time_serials[monitor_window_length+target_skip_steps:monitor_window_length+target_skip_steps+plot_length, i]).reshape(-1, 1), label='target', c='b', marker='.')
        plt.title(f'{feature_names[i]}')

    # sorted_outputs = np.array(list(map(list, zip(*flat_results[:plot_length])))[0])
    # sorted_targets = np.array(list(map(list, zip(*flat_results[:plot_length])))[1])
    # for i in range(feature_size):
    #     plt.subplot(4, math.ceil(feature_size / 2), feature_size + i + 1)
    #     plt.scatter(x, sorted_outputs[:, i].reshape(1, -1), label='predict', c='r', marker='+')
    #     plt.scatter(x, sorted_targets[:, 0, i].reshape(1, -1), label='target', c='b', marker='x')
    #     plt.title(f'{feature_names[i]}')
    for i in range(1, 10):
        file_name = f"mul-lstm-transformer-attn-res{res}-{i}.png"
        if not os.path.exists(os.path.join(ROOT_DIR, "results", file_name)):
            plt.savefig(os.path.join(ROOT_DIR, "results", file_name))
            # plt.show()
            break

# multi_time_serial_lstm_transformer_attention(monitor_window_length=50, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=[])
multi_time_serial_lstm_transformer_attention(monitor_window_length=200, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=[])
multi_time_serial_lstm_transformer_attention(monitor_window_length=50, window_sample_interval=10, target_skip_steps=4, batch_size=128, epochs_num=20, feature_names=[])
multi_time_serial_lstm_transformer_attention(monitor_window_length=200, window_sample_interval=10, target_skip_steps=9, batch_size=128, epochs_num=20, feature_names=[])
exit(0)
multi_time_serial_lstm_transformer_attention(monitor_window_length=50, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])
multi_time_serial_lstm_transformer_attention(monitor_window_length=200, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])
multi_time_serial_lstm_transformer_attention(monitor_window_length=50, window_sample_interval=10, target_skip_steps=4, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])
multi_time_serial_lstm_transformer_attention(monitor_window_length=200, window_sample_interval=10, target_skip_steps=9, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])

multi_time_serial_lstm(monitor_window_length=50, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])
multi_time_serial_lstm(monitor_window_length=200, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])
multi_time_serial_lstm(monitor_window_length=50, window_sample_interval=10, target_skip_steps=4, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])
multi_time_serial_lstm(monitor_window_length=200, window_sample_interval=10, target_skip_steps=9, batch_size=128, epochs_num=20, feature_names=['CAN/EngineSpeed_CAN', 'CAN/VehicleSpeed', '/Plugins/Velocity_X'])
multi_time_serial_lstm(monitor_window_length=50, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=[])
multi_time_serial_lstm(monitor_window_length=200, window_sample_interval=10, target_skip_steps=0, batch_size=128, epochs_num=20, feature_names=[])
multi_time_serial_lstm(monitor_window_length=50, window_sample_interval=10, target_skip_steps=4, batch_size=128, epochs_num=20, feature_names=[])
multi_time_serial_lstm(monitor_window_length=200, window_sample_interval=10, target_skip_steps=9, batch_size=128, epochs_num=20, feature_names=[])

# for i in range(4):
#     print(i+1)

#     except Exception as e:
#         continue
